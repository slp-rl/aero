"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""
import itertools
import logging
import os
import shutil

import hydra
import wandb

from src.ddp.executor import start_ddp_workers
from src.models import modelFactory
from src.utils import print_network
from src.wandb_logger import _init_wandb_run

logger = logging.getLogger(__name__)


def run(args):
    import torch

    from src.ddp import distrib
    from src.data.datasets import LrHrSet
    from src.solver import Solver
    logger.info(f'calling distrib.init')
    
    distrib.init(args)

    _init_wandb_run(args)

    if distrib.rank == 0:
        if os.path.exists(args.samples_dir):
            shutil.rmtree(args.samples_dir)
        os.makedirs(args.samples_dir)

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)

    models = modelFactory.get_model(args)
    for model_name, model in models.items():
        print_network(model_name, model, logger)
    wandb.watch(tuple(models.values()), log=args.wandb.log, log_freq=args.wandb.log_freq)

    if args.show:
        logger.info(models)
        mb = sum(p.numel() for p in models.parameters()) * 4 / 2 ** 20
        logger.info('Size: %.1f MB', mb)
        return

    assert args.experiment.batch_size % distrib.world_size == 0
    args.experiment.batch_size //= distrib.world_size

    channels = 1
    if args.experiment.model == 'aero':
       channels = args.experiment.aero.in_channels
    elif args.experiment.model == 'seanet':
       channels = args.experiment.seanet.in_channels

    # Building datasets and loaders
    tr_dataset = LrHrSet(args.dset.train, args.experiment.lr_sr, args.experiment.hr_sr,
                         args.experiment.stride, args.experiment.segment, upsample=args.experiment.upsample,
                         channels=channels, vary_volume=args.vary_volume)
    tr_loader = distrib.loader(tr_dataset, batch_size=args.experiment.batch_size, shuffle=True,
                               num_workers=args.num_workers)

    if args.dset.valid:
        args.valid_equals_test = args.dset.valid == args.dset.test

    if args.dset.valid:
        cv_dataset = LrHrSet(args.dset.valid, args.experiment.lr_sr, args.experiment.hr_sr,
                            stride=None, segment=None, upsample=args.experiment.upsample,
                            channels=channels)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        cv_loader = None

    if args.dset.test:
        tt_dataset = LrHrSet(args.dset.test, args.experiment.lr_sr, args.experiment.hr_sr,
                             stride=None, segment=None, with_path=True, upsample=args.experiment.upsample,
                             channels=channels)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    if torch.cuda.is_available() and args.device=='cuda':
        for model in models.values():
            model.cuda()

    # optimizer
    optimizers = {}
    if args.optim == "adam":
        if 'adversarial' in args.experiment and args.experiment.adversarial:
            optimizer = torch.optim.Adam(
                itertools.chain(models['generator'].parameters())
                , lr=args.lr, betas=(args.beta1, args.beta2))
            disc_lr = args.disc_lr if 'disc_lr' in args else args.lr
            disc_optimizer = torch.optim.Adam(
                itertools.chain(*[models[disc_name].parameters() for disc_name in
                                args.experiment.discriminator_models]),
                disc_lr, betas=(args.beta1, args.beta2),maximize=True)
            optimizers.update({'optimizer': optimizer})
            optimizers.update({'disc_optimizer': disc_optimizer})
        else:
            optimizer = torch.optim.Adam(models['generator'].parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            optimizers.update({'optimizer': optimizer})
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    # Construct Solver
    solver = Solver(data, models, optimizers, args)
    solver.train()

    distrib.close()



def _main(args):
    global __file__
    print(args)
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str):
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)



    if args.ddp and args.rank is None:
        start_ddp_workers(args)
    else:
        run(args)

    wandb.finish()


@hydra.main(config_path="conf", config_name="main_config", version_base="1.1")  # for latest version of hydra=1.0
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
