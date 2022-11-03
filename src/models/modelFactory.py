from src.models.aero import Aero
from src.models.discriminators import Discriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator


def get_model(args):
    if args.experiment.model == 'aero':
        generator = Aero(**args.experiment.aero)

    models = {'generator': generator}

    if 'adversarial' in args.experiment and args.experiment.adversarial:
        if 'msd_melgan' in args.experiment.discriminator_models:
            discriminator = Discriminator(**args.experiment.melgan_discriminator)
            models.update({'msd_melgan': discriminator})
        if 'msd_hifi' in args.experiment.discriminator_models:
            msd = MultiScaleDiscriminator(**args.experiment.msd)
            models.update({'msd': msd})
        if 'mpd' in args.experiment.discriminator_models:
            mpd = MultiPeriodDiscriminator(**args.experiment.mpd)
            models.update({'mpd': mpd})
        if 'hifi' in args.experiment.discriminator_models:
            mpd = MultiPeriodDiscriminator(**args.experiment.mpd)
            msd = MultiScaleDiscriminator(**args.experiment.msd)
            models.update({'mpd': mpd, 'msd': msd})

    return models