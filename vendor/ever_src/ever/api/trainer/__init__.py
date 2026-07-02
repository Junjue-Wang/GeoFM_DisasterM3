import argparse
import os
import shutil

from .th_ddp_trainer import THDDPTrainer
from .trainer import Trainer
from .pseudo_labeling_trainer import PseudoLabelTrainer

TRAINER = dict(
    th_ddp=THDDPTrainer,
    base=Trainer,
    pseudo_labeling_trainer=PseudoLabelTrainer
)


def get_default_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', default=None, type=str,
                        help='path to config file')
    parser.add_argument('--model_dir', default=None, type=str,
                        help='path to model directory')
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument('--trainer', default='th_ddp', type=str,
                        help='type of trainer')
    parser.add_argument('--find_unused_parameters', default=False, type=bool,
                        help='whether to find unused parameters')
    parser.add_argument('--amp', action='store_true',
                        help='whether to use automatic mixed precision (amp) training')

    # command line options
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def initialize_workspace(config_path, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    if config_path.endswith('.py'):
        shutil.copy(config_path,
                    os.path.join(model_dir, 'config.py'))
    else:
        cfg_path_segs = ['configs'] + config_path.split('.')
        cfg_path_segs[-1] = cfg_path_segs[-1] + '.py'
        shutil.copy(os.path.join(os.path.curdir, *cfg_path_segs),
                    os.path.join(model_dir, 'config.py'))


def get_trainer(trainer_name=None, parser=None, return_args=False):
    if parser is None:
        parser = get_default_parser()
    args = parser.parse_args()
    # check args
    assert args.config_path is not None, 'The `config_path` is needed'
    assert args.model_dir is not None, 'The `model_dir` is needed'

    # initialize directory
    initialize_workspace(args.config_path, args.model_dir)

    # compatible with torchrun and torch.distributed.launch
    if args.local_rank is None:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if trainer_name is None:
        trainer_name = args.trainer

    if trainer_name == 'th_fsdp':
        from .th_fsdp_trainer import THFSDPTrainer
        TRAINER.update(dict(th_fsdp=THFSDPTrainer))

    if return_args:
        return TRAINER[trainer_name](args), args
    else:
        return TRAINER[trainer_name](args)
