"""
watchmal.py

Base module with common global variables to be used by model-specific main scripts
"""

# Standard python imports
from os import popen

# WatChMaL imports
from io_utils.ioconfig import load_config, save_config, to_kwargs, ConfigAttr
from io_utils.arghandler import Argument, parse_args
from io_utils.modelhandler import check_params, select_model

# List of arguments to request from commandline
_ARGS=[Argument('model', list, '-mdl', list_dtype=str, default=['resnet', 'resnet18'],
                help='Specify model architecture. Default = resnet resnet18.'),
       Argument('model_params', list, '-pms', list_dtype=str, default=None,
                help='Soecify model constructer args. Default=None.'),
       Argument('itp_params', list, '-itp', list_dtype=str, default=None,
                help='Specify interpolatation args. Default=None.'),
       Argument('device', str, '-dev', default='cpu', help='Choose either cpu or gpu, Default=cpu.'),
       Argument('gpu_list', list, '-gpu', list_dtype=int, default=0,
                help='Indices of the device GPUs to utilize. E.g. 0 1. Default=0.'),
       Argument('path', str, '-pth', default=None, help='Absolute path for the training dataset. Default=None.'),
       Argument('cl_ratio', float, '-clr', default=0.1,
                help='Fraction of dataset to be used for classifier training and validation. Default=0.1.'),
       Argument('val_indices', str, '-vif', default=None,
                help='Validation indices file Default=None'),
       Argument('test_indices', str, '-tif', default=None,
                help='Test indices file Default=None'),
       Argument('train_indices', str, '-trif', default=None,
                help='Train indices file Default=None'),
       Argument('epochs', float, '-epc', default=10.0, help='Number of training epochs. Default=10.0'),
       Argument('batch_size_train', int, '-btn', default=128, help='Training dataset batch size. Default=128.'),
       Argument('batch_size_val', int, '-bvl', default=1000, help='Batch size for validation.'),
       Argument('batch_size_test', int, '-btt', default=1000, help='Batch size for testing.'),
       Argument('tasks', list, list_dtype=str, flag='-d', default=['train', 'test', 'valid'],
                help='Specify list of tasks to be performed in this run.'),
       Argument('dump_path', str, '-dmp', default='dumps/', help='Specify path to dump data to. Default=dumps.'),
       Argument('load', str, '-lod', default=None, help='Specify config file to load from. No action by default.'),
       Argument('restore_state', str, '-ret', default=None, help='Specify a model state file to restore from.'),
       Argument('cfg', str, '-sav', default=None,
                help='Specify name for destination config file. No action by default.'),
       Argument('githash', str, '-git', default=None, help='git-hash for the latest commit'),
       Argument('report_interval', int, '-rpt', default=10,
                help='Interval at which to report the training metrics to the user'),
       Argument('num_vals', int, '-nvl', default=1000,
                help='Number of validations to perform during the entire training'),
       Argument('num_val_batches', int, '-nvb', default=16,
                help='Number of batches to use for each validation during training'),
       Argument('num_dump_events', int, '-nde', default=128,
                help='Number of events and their results to dump to a .npz file'),
       Argument('lr', float, '-lr', default=0.0001, help='Initial learning rate to use for the optimizer.'),
       Argument('train_all', int, '-tal', default=0, help='Used for modular and pre-trained models.')]

_ATTR_DICT={arg.name: ConfigAttr(arg.name, arg.dtype, list_dtype=arg.list_dtype if hasattr(arg, 'list_dtype') else None) for arg in _ARGS}


def handle_config():
    """Parse the arguments and setup the config object."""

    # Parse the arguments specified by cmdline
    config=parse_args(_ARGS)

    # Set overwrite to False for attributes specified by cmdline
    for ar in _ARGS:
        if getattr(config, ar.name) != ar.default:
            _ATTR_DICT[ar.name].overwrite=False

    # Load the config from file
    if config.load is not None:
        load_config(config, config.load, _ATTR_DICT)

    # Add the git-hash from the latest commit to config
    git_hash=popen("git rev-parse HEAD").read()
    config.githash=git_hash[:len(git_hash) - 1]

    # Set save directory to under USER_DIR
    config.dump_path=config.dump_path + ('' if config.dump_path.endswith('/') else '/')

    # Save the config object to file
    if config.cfg is not None:
        save_config(config, config.cfg)

    return config


def handle_model(model_id, model_params):
    """Call the appropriate model constructor given config args."""

    # Check if the constructor params are valid
    check_params(model_id[0], to_kwargs(model_params))

    # Choose the appropriate constructor and create the model object
    constructor=select_model(model_id)
    return constructor(**to_kwargs(model_params))
