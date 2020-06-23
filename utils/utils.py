import tensorflow.compat.v1 as tf
import os
import pandas as pd
from t5.evaluation import eval_utils

# Same from pretraing code
# "Pre-training" ==> FineTuning
DICT_BATCH_SIZE_PRETRAIN = {
        'small': 256,
        'base': 128,
        'large': 64
        }
BRWAC_TXT_LEN = 7361359

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

