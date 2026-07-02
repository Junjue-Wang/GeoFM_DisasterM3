import torch
import torch.distributed as dist
import torch.nn as nn

from ever.api.trainer import trainer
from ever.core.pseudo_label_launcher import PseudoLabelLauncher
from ever.core.builder import make_dataloader
from ever.core.builder import make_optimizer
from .trainer import merge_dict
from ever.util import param_util

class PseudoLabelTrainer(trainer.Trainer):
    def __init__(self, args):
        super().__init__(args)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://"
            )

    def make_model(self):
        model = super(PseudoLabelTrainer, self).make_model()
        if self.config.train.get('sync_bn', False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(self.device)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=self.args.find_unused_parameters,
        )
        return model


    def make_dataloader(self):
        sourcedata_loader = make_dataloader(self.config.data.source)
        targetdata_loader = make_dataloader(self.config.data.target)
        targetdata_infer_loader = make_dataloader(self.config.data.target_infer)
        testdata_loader = make_dataloader(self.config.data.test) if 'test' in self.config.data else None

        return dict(traindata_loader=sourcedata_loader, testdata_loader=testdata_loader, targetdata_loader=targetdata_loader, targetdata_infer_loader=targetdata_infer_loader)

    def build_launcher(self):
        kwargs = dict(model_dir=self.args.model_dir)
        kwargs.update(dict(model=self.make_model().to(self.device)))
        kwargs.update(self.make_lr_optimizer(kwargs['model'].module.custom_param_groups()))
        kwargs.update(dict(amp=self.args.amp))
        kwargs.update(dict(config=self.config))
        tl = PseudoLabelLauncher(**kwargs)

        return dict(config=self.config, launcher=tl)

    def run(self, after_construct_launcher_callbacks=None):
        tl = self.build_launcher()['launcher']
        kw_dataloader = self.make_dataloader()
        param_util.trainable_parameters(tl.model, tl.logger)
        param_util.count_model_parameters(tl.model, tl.logger)

        for c in self._callbacks:
            tl.logger.info(f'callback: {c}')
            tl.register_callback(c)

        if after_construct_launcher_callbacks is not None:
            for f in after_construct_launcher_callbacks:
                f(tl)

        tl.logger.info('th sync bn: {}'.format(
            'True' if self.config.train.get('sync_bn', False) else 'False'))
        tl.logger.info('external parameter: {}'.format(self.args.opts))

        # start training
        tl.train_by_config(kw_dataloader['traindata_loader'],
                           config=merge_dict(self.config.train, self.config.test),
                           test_data_loader=kw_dataloader['testdata_loader'],
                           target_data_loader=kw_dataloader['targetdata_loader'],
                           targetdata_infer_loader=kw_dataloader['targetdata_infer_loader']
                           )

        return dict(config=self.config, launcher=tl)
