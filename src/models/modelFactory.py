from src.models.aero import Aero
from src.models.discriminators import Discriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator


def get_model(args):
    if args.experiment.model == 'aero':
        generator = Aero(**args.experiment.aero)


    models = {'generator': generator}

    if 'adversarial' in args.experiment and args.experiment.adversarial:
        multiple_discriminators_mode = 'multiple_discriminators' in args.experiment and args.experiment.multiple_discriminators
        if multiple_discriminators_mode:
            if 'melgan' in args.experiment.discriminator_models:
                discriminator = Discriminator(**args.experiment.melgan_discriminator)
                models.update({'melgan': discriminator})
            if 'msd' in args.experiment.discriminator_models:
                msd = MultiScaleDiscriminator(**args.experiment.msd)
                models.update({'msd': msd})
            if 'mpd' in args.experiment.discriminator_models:
                mpd = MultiPeriodDiscriminator(**args.experiment.mpd)
                models.update({'mpd': mpd})
            if 'hifi' in args.experiment.discriminator_models:
                mpd = MultiPeriodDiscriminator(**args.experiment.mpd)
                msd = MultiScaleDiscriminator(**args.experiment.msd)
                models.update({'mpd': mpd, 'msd': msd})
            if 'stft' in args.experiment.discriminator_models:
                discriminator = STFTDiscriminator(**args.experiment.stft_discriminator)
                models.update({'stft_disc': discriminator})
            if 'mbd' in args.experiment.discriminator_models:
                mbd = MultiPeriodDiscriminator(**args.experiment.mbd)
                models.update({'mbd': mbd})
            if 'spec' in args.experiment.discriminator_models:
                spec_discriminator = SpecDiscriminator(**args.experiment.spec_discriminator)
                models.update({'spec': spec_discriminator})
        else:
            if args.experiment.discriminator_model == 'melgan':
                discriminator = Discriminator(**args.experiment.melgan_discriminator)
                models.update({'melgan':discriminator})
            elif args.experiment.discriminator_model == 'msd':
                msd = MultiScaleDiscriminator(**args.experiment.msd)
                models.update({'msd': msd})
            elif args.experiment.discriminator_model == 'mpd':
                mpd = MultiPeriodDiscriminator(**args.experiment.mpd)
                models.update({'mpd': mpd})
            elif args.experiment.discriminator_model == 'hifi':
                mpd = MultiPeriodDiscriminator(**args.experiment.mpd)
                msd = MultiScaleDiscriminator(**args.experiment.msd)
                models.update({'mpd': mpd, 'msd': msd})
            elif args.experiment.discriminator_model == 'stft':
                discriminator = STFTDiscriminator(**args.experiment.stft_discriminator)
                models.update({'stft_disc': discriminator})
            elif args.experiment.discriminator_model == 'mbd':
                mbd = MultiBandDiscriminator(**args.experiment.mbd)
                models.update({'mbd': mbd})
            elif args.experiment.discriminator_model == 'spec':
                spec_discriminator = SpecDiscriminator(**args.experiment.spec_discriminator)
                models.update({'spec': spec_discriminator})
    return models