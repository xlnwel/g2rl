import tensorflow as tf

from utility.display import pwc


def restore(ckpt_manager, ckpt, ckpt_path, name='model'):
    """ Restores the latest parameter recorded by ckpt_manager

    Args:
        ckpt_manager: An instance of tf.train.CheckpointManager
        ckpt: An instance of tf.train.Checkpoint
        ckpt_path: The directory in which to write checkpoints
        name: optional name for print
    """
    path = ckpt_manager.latest_checkpoint
    if path:
        ckpt.restore(path)#.assert_consumed()
        pwc(f'Params for {name} are restored from "{path}".', color='cyan')
    else:
        pwc(f'No model for {name} is found at "{ckpt_path}"!', color='cyan')
    return bool(path)

def save(ckpt_manager, print_terminal_info=True):
    """ Saves model

    Args:
        ckpt_manager: An instance of tf.train.CheckpointManager
        message: optional message for print
    """
    path = ckpt_manager.save()
    if print_terminal_info:
        pwc(f'Model saved at {path}', color='cyan')

def setup_checkpoint(ckpt_models, root_dir, model_name, name='model', **kwargs):
    """ Setups checkpoint

    Args:
        ckpt_models: A dict of models to save, including optimizers
        root_dir: The root directory for checkpoint
    """
    if not model_name:
        model_name = 'baseline'
    # checkpoint & manager
    ckpt = tf.train.Checkpoint(**ckpt_models, **kwargs)
    ckpt_path = f'{root_dir}/{model_name}/{name}'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, 5)
    
    return ckpt, ckpt_path, ckpt_manager
