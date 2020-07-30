import collections
import os
import pandas as pd
import tensorflow.compat.v1 as tf

# Same from pretraing code
DICT_BATCH_SIZE_PRETRAIN = {'small': 256, 'base': 128, 'large': 64}

# Same from pretraing code
BRWAC_TXT_LEN = 7361359


# Same from pretraing code
def epoch_to_steps(batch_size, epochs, total_examples=BRWAC_TXT_LEN):
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

# Extracted from https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/evaluation/eval_utils.py
# slightly modified to also extract wall time from tensorboard events logs
def parse_events_files(tb_summary_dir):
    """Parse all TensorBoard events files in tb_summary_dir.
    Args:
        tb_summary_dir: str, path to look for events files in.
    Returns:
        A dict, where each key is a TensorBoard tag and each value is a list of
        Event tuples with step and value attributes.
  """
    Event = collections.namedtuple("event", ["step", "value", "wall_time"])
    events = collections.defaultdict(list)
    for events_file in tf.gfile.Glob(os.path.join(tb_summary_dir, "events.*")):
        try:
            for e in tf.train.summary_iterator(events_file):
                for v in e.summary.value:
                    events[v.tag].append(
                        Event(e.step, v.simple_value, e.wall_time))
        except tf.errors.DataLossError:
            logging.warning("Skipping %s due to truncated record.", events_file)
    return events


def tf_events_to_pandas(tb_summary_dir, tag='loss'):
    """Parse tensorboard logs into a padas dataframe

    Args:
        tb_summary_dir: Path to search for events.* files
        tag: tag to extract from logs

    Returns:
        pandas.DataFrame, containing two pandas.Series: steps and tag
    """
    events = parse_events_files(tb_summary_dir)
    df = pd.DataFrame({
        'step': [x.step for x in events[tag]],
        'wall_time': [x.wall_time for x in events[tag]],
        tag: [x.value for x in events[tag]]
    }).sort_values(by='step')
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

    return steps / epoch_to_steps(batch_size, 1, total_examples=total_examples)


def smooth_array(input, smooth):
    """Smooth array using exponential moving average

    Args:
        input: input data
        smooth: smooth factor, 0<=smooth<1

    Returns:
        Smoothed array
    """
    return input.ewm(alpha=(1 - smooth)).mean()
