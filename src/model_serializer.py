import os
from pathlib import Path
import logging

import torch

from src.utils import copy_state

logger = logging.getLogger(__name__)

SERIALIZE_KEY_MODELS = 'models'
SERIALIZE_KEY_OPTIMIZERS = 'optimizers'
SERIALIZE_KEY_HISTORY = 'history'
SERIALIZE_KEY_STATE = 'state'
SERIALIZE_KEY_BEST_STATES = 'best_states'
SERIALIZE_KEY_ARGS = 'args'


def serialize_model(model):
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


def _serialize_models(models):
    serialized_models = {}
    for name, model in models.items():
        serialized_models[name] = serialize_model(model)
    return serialized_models


def _serialize_optimizers(optimizers):
    serialized_optimizers = {}
    for name, optimizer in optimizers.items():
        serialized_optimizers[name] = optimizer.state_dict()
    return serialized_optimizers


def serialize(models, optimizers, history, best_states, args, save_latest_generator=True):
    checkpoint_file = Path(args.checkpoint_file)
    best_file = Path(args.best_file)

    package = {}
    package[SERIALIZE_KEY_MODELS] = _serialize_models(models)
    package[SERIALIZE_KEY_OPTIMIZERS] = _serialize_optimizers(optimizers)
    package[SERIALIZE_KEY_HISTORY] = history
    package[SERIALIZE_KEY_BEST_STATES] = best_states
    package[SERIALIZE_KEY_ARGS] = args
    tmp_path = str(checkpoint_file) + ".tmp"
    torch.save(package, tmp_path)
    # renaming is sort of atomic on UNIX (not really true on NFS)
    # but still less chances of leaving a half written checkpoint behind.
    os.replace(tmp_path, checkpoint_file)

    if save_latest_generator:
        generatorPackage = {}
        generatorPackage[SERIALIZE_KEY_STATE]=package[SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE]
        model_filename = "generator_latest.th"
        tmp_path = os.path.join(checkpoint_file.parent, model_filename) + ".tmp"
        torch.save(generatorPackage, tmp_path)
        model_path = Path(checkpoint_file.parent / model_filename)
        os.replace(tmp_path, model_path)

    # Saving only the latest best model.
    models = package[SERIALIZE_KEY_MODELS]
    for model_name, best_state in package[SERIALIZE_KEY_BEST_STATES].items():
        models[model_name][SERIALIZE_KEY_STATE] = best_state
        model_filename = model_name + '_' + best_file.name
        tmp_path = os.path.join(best_file.parent, model_filename) + ".tmp"
        torch.save(models[model_name], tmp_path)
        model_path = Path(best_file.parent / model_filename)
        os.replace(tmp_path, model_path)
