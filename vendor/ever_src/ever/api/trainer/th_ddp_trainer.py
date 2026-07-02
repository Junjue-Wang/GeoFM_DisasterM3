import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn

from ever.api.trainer import trainer
from ever.core.launcher import Launcher


class THDDPTrainer(trainer.Trainer):
    def __init__(self, args):
        super().__init__(args)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)
            # Generous collective timeout: slow in-training eval (e.g. sliding-window
            # full-tile val) can stall >NCCL default 10min watchdog → spurious
            # ALLREDUCE timeout. Override via env GEOFM_NCCL_TIMEOUT_SEC (default 7200s).
            _to = int(os.environ.get("GEOFM_NCCL_TIMEOUT_SEC", "7200"))
            dist.init_process_group(
                backend="nccl", init_method="env://",
                timeout=timedelta(seconds=_to),
            )

    def make_model(self):
        model = super(THDDPTrainer, self).make_model()
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

    def build_launcher(self):
        kwargs = dict(model_dir=self.args.model_dir, amp=self.args.amp)
        kwargs.update(dict(model=self.make_model()))
        kwargs.update(
            self.make_lr_optimizer(kwargs['model'].module.custom_param_groups()))
        tl = Launcher(**kwargs)

        return dict(config=self.config, launcher=tl)
