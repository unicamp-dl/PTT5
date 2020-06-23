import tensorflow.compat.v1 as tf
import os
import pandas as pd
from t5.evaluation import eval_utils

# Same from pretraing code
DICT_BATCH_SIZE_PRETRAIN = {
        'small': 256,
        'base': 128,
        'large': 64
        }

# Same from pretraing code
BRWAC_TXT_LEN = 7361359

# Same from pretraing code
def epoch_to_steps(batch_size,epochs, total_examples=BRWAC_TXT_LEN):
    """Calculates number of steps

    Args:
        batch_size: batch size
        epochs: number of epochs
        total_examples: total data size

    Returns:
        Number of steps
    """
    return int((epochs * total_examples) // batch_size)

def get_model_size_from_dir(tb_summary_dir):
    """Return model size from dir

    Args:
        tb_summary_dir: path for tensorboard logs, last path should follow
        the structutre <size>_...

    Returns:
        Model size

    """
    return os.path.basename(os.path.normpath(tb_summary_dir)).split('_')[0]

def tf_events_to_pandas(tb_summary_dir, tag='loss'):
    """Parse tensorboard logs into a padas dataframe

    Args:
        tb_summary_dir: Path to search for events.* files
        tag: tag to extract from logs

    Returns:
        pandas.DataFrame, containing two pandas.Series: steps and tag
    """
    events = eval_utils.parse_events_files(tb_summary_dir)
    df = pd.DataFrame({
        'step': [x.step for x in events[tag]],
        tag: [x.value for x in events[tag]]
        })
    return df

def fix_step_offset(steps):
    """Fix steps offset, returning array starting from zero

    Args:
        steps: array with steps

    Returns;
        steps array starting from zero
    """
    return steps - steps.min()

def step_to_epoch(steps, batch_size, total_examples=BRWAC_TXT_LEN):
    """Convert array with step values to array of epoch values

    Args:
        steps: pd.Series or np.array containing step values
        batch_size: batch size
        total_examples: total data size

    Returns:
        Array with epoch values
    """

    return steps / epoch_to_steps(batch_size,1,total_examples=total_examples)
