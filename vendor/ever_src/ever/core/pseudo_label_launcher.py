from .launcher import Launcher, AttrDict
from ever.interface.callback import SaveCheckpointCallback
from .iterator import get_iterator
from ever.interface.callback import EvaluationCallback
from ever.interface.callback import Callback
import types, time, torch
from . import to

class GeneratePseudoLabelCallBack(Callback):
    def __init__(self,
                 dataloader,
                 epoch_interval: int,
                 only_master: bool,
                 after_train=True,
                 config=None):
        super().__init__(epoch_interval=epoch_interval, only_master=only_master,
                         before_train=False,
                         after_train=after_train)

        self._dataloader = dataloader
        self._config = config

    def func(self):
        self._dataloader = self.launcher.generate_pl(self._dataloader, config=self._config)

    def name(self):
        return 'GeneratePseudoLabel'


class PseudoLabelLauncher(Launcher):
    def __init__(self,
                 model_dir,
                 model,
                 optimizer,
                 lr_schedule,
                 amp,
                 config=None
                 ):
        super(PseudoLabelLauncher, self).__init__(model_dir,
                                                  model,
                                                  optimizer,
                                                  lr_schedule,
                                                  amp)
        self.config = config
        self._pseudo_label_prepared = self.config.train.get('pseudo_label_prepared', False)

    def generate_pl(self, data_loader, config=None):
        if not self._training:
            self.init()
        return self._generate_pl_fn(data_loader, config)

    def _generate_pl_fn(self, data_loader, config=None):
            raise NotImplementedError

    def override_generate_pl(self, fn):
        self._generate_pl_fn = types.MethodType(fn, self)


    def train_by_config(self, train_data_loader, config, test_data_loader=None, target_data_loader=None, targetdata_infer_loader=None):
        self._training = True
        if config.get('resume_from_last', True):
            self.init()
        self._model.train()
        forward_times = 1

        if self._master:
            self._logger.equation('batch_size_per_gpu',
                                  train_data_loader.batch_sampler.batch_size)
            self._logger.forward_times(forward_times)
            self._logger.approx_equation('num_epochs',
                                         round(
                                             config['num_iters'] * forward_times / len(
                                                 train_data_loader), 1))
            self._logger.equation('num_iters', config['num_iters'])
            self._logger.equation('optimizer', self.optimizer)

            model_extra_info = self.er_model.log_info()
            model_extra_info['model.type'] = self.er_model.__class__.__name__

            for k, v in model_extra_info.items():
                self._logger.equation(k, v)

        signal_loss_dict = self.train_multi_stage_iters(train_data_loader,
                                            test_data_loader=test_data_loader, target_data_loader=target_data_loader, targetdata_infer_loader=targetdata_infer_loader, **config)

        return signal_loss_dict

    def train_multi_stage_iters(self,
                    train_data_loader,
                    test_data_loader=None,
                    target_data_loader=None,
                    targetdata_infer_loader=None,
                    **kwargs):
        distributed = kwargs.get('distributed', False)

        num_iters = kwargs.get('num_iters', -1)
        forward_times = 1

        eval_per_epoch = kwargs.get('eval_per_epoch', False)
        eval_interval_epoch = kwargs.get('eval_interval_epoch', -1)
        eval_after_train = kwargs.get('eval_after_train', False)

        tensorboard_interval_step = kwargs.get('tensorboard_interval_step', 100)
        log_interval_step = kwargs.get('log_interval_step', 1)
        log_model_dir_interval_step = kwargs.get('task_log_interval_step', 500)

        summary_grads = kwargs.get('summary_grads', False)
        summary_weights = kwargs.get('summary_weights', False)

        iterator_type = kwargs.get('iterator_type', 'normal')

        save_ckpt_interval_epoch = kwargs.get('save_ckpt_interval_epoch', 1)

        dist_eval = kwargs.get('distributed_evaluate', False)

        generate_interval_step = kwargs.get('generate_interval_step', -1)
        warm_up_step = kwargs.get('warm_up_step', -1)
        pseudo_loss_weight = kwargs.get('pseudo_loss_weight', 0.5)

        iterator = get_iterator(iterator_type)(train_data_loader)
        target_iterator = get_iterator(iterator_type)(target_data_loader)
        self.register_callback(SaveCheckpointCallback(save_ckpt_interval_epoch))

        if eval_per_epoch or eval_after_train:
            if eval_per_epoch and eval_interval_epoch < 0:
                raise ValueError(
                    'eval_interval_epoch should be a positive number when eval_per_epoch = True')
            if not eval_per_epoch and eval_interval_epoch > 0:
                raise ValueError(
                    'eval_per_epoch should be True when eval_interval_epoch > 0')

            self.register_callback(
                EvaluationCallback(test_data_loader, eval_interval_epoch, not dist_eval,
                                   config=AttrDict.from_dict(kwargs),
                                   after_train=eval_after_train)
            )
        self._callbacks.sort(key=lambda callback: callback.prior)

        self.run_callbacks('before_train')

        signal_loss_dict = dict()
        while self._ckpt.global_step < num_iters:
            start = time.time()
            if distributed:
                iterator.set_seed_for_dist_sampler(self._ckpt.global_step)

            if self._ckpt.global_step >= warm_up_step and self._ckpt.global_step % generate_interval_step == 0:
                # use infer target loader to update training target loader
                target_data_loader = self.generate_pl(targetdata_infer_loader, config=self.config)
                target_iterator = get_iterator(iterator_type)(target_data_loader)
                self._pseudo_label_prepared = True

            with torch.autograd.profiler.record_function('load_data'):
                data_list = iterator.next(forward_times,
                                          call_backs=self._callbacks,
                                          is_master=self._master)
                if self._pseudo_label_prepared:
                    if distributed:
                        target_iterator.set_seed_for_dist_sampler(self._ckpt.global_step)
                    target_data_list = target_iterator.next(forward_times)
                    target_data = to.to_device(target_data_list, self._device)[0]

            data_time = time.time() - start
            self._model.train()

            data = to.to_device(data_list, self._device)[0]

            with torch.autograd.profiler.record_function('forward_backward'):
                msg_dict = self.compute_loss_gradient(data)
                if self._pseudo_label_prepared:
                    pse_msg_dict = dict()
                    for k, v in self.compute_loss_gradient(target_data).items():
                        pse_msg_dict[f'{pseudo_loss_weight}@pse_'+k] = v * pseudo_loss_weight
                    msg_dict.update(pse_msg_dict)

            msg_dict = self.log_info_dict(msg_dict)
            signal_loss_dict = msg_dict.copy()

            if self._master:
                if summary_grads and self._ckpt.global_step % tensorboard_interval_step == 0:
                    self._logger.summary_grads(module=self.er_model,
                                               step=self._ckpt.global_step)

            with torch.autograd.profiler.record_function('update_lr_params'):
                self.update_training_status()

            if self._master:
                time_cost = time.time() - start
                epoch = iterator._current_epoch

                self._logger.train_log(step=self._ckpt.global_step,
                                       epoch=epoch,
                                       loss_dict=msg_dict,
                                       data_time=data_time,
                                       time_cost=time_cost,
                                       lr=self.lr,
                                       num_iters=num_iters,
                                       tensorboard_interval_step=tensorboard_interval_step,
                                       log_interval_step=log_interval_step)
                if (log_model_dir_interval_step > 0) and (
                        self._ckpt.global_step % log_model_dir_interval_step == 0):
                    self._logger.info(self.model_dir)

                if summary_weights and self._ckpt.global_step % tensorboard_interval_step == 0:
                    self._logger.summary_weights(module=self.er_model,
                                                 step=self._ckpt.global_step)

        self.run_callbacks('after_train')
        return signal_loss_dict