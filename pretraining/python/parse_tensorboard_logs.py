import sys
sys.path.append('../../utils/')
import utils
import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import OrderedDict

LOGS_DIR = '../logs_tensorboard'

# Starting steps for each model size - from google GCS
initial_step_model_size = {
    'small': 1_000_000,
    'base': 999_900,
    'large': 1_000_700
}

plt.rcParams.update({'font.size': 16})

# Experiments names - order is important to generate plots correctly
exp_names = [
    # Main initial experiments - all weights are updated
    'small_standard_vocab',
    'small_custom_sentencepiece_vocab',
    'base_standard_vocab',
    'base_custom_sentencepiece_vocab',
    'large_standard_vocab',
    'large_custom_sentencepiece_vocab',
    # Only embeddings are updated
    'small_embeddings_only_standard_vocab',
    'small_embeddings_only_custom_sentencepiece_vocab',
    'base_embeddings_only_standard_vocab',
    'base_embeddings_only_custom_sentencepiece_vocab',
    'large_embeddings_only_standard_vocab',
    'large_embeddings_only_custom_sentencepiece_vocab',
    # Double batch size for large (128 = 64 * 2)
    'large_batchsize_128_custom_sentencepiece_vocab',
    'large_batchsize_128_standard_vocab',
]


def parse_logs() -> OrderedDict[str, pd.DataFrame]:
    """Parses tensorboard logs from pretraining experiments

    Returns:
        OrderedDict[str, pd.DataFrame]: dict of dataframes, each one containing
        data from one experiment
    """
    dict_dfs_events = OrderedDict()
    for name in tqdm.tqdm(exp_names,
                          desc='Reading tensorboard logs into pandas'):
        model_size = utils.get_model_size_from_dir(name)
        if name.startswith('large_batchsize_128'):
            batch_size = 128
        else:
            batch_size = utils.DICT_BATCH_SIZE_PRETRAIN[model_size]
        dict_dfs_events[name] = utils.tf_events_to_pandas(
            os.path.join(LOGS_DIR, name))
        dict_dfs_events[name]['fixed_step'] = utils.fix_step_offset(
            dict_dfs_events[name]['step'])
        dict_dfs_events[name]['fixed_step'] = dict_dfs_events[name][
            'step'] - initial_step_model_size[model_size]
        dict_dfs_events[name]['epoch'] = utils.step_to_epoch(
            dict_dfs_events[name]['fixed_step'], batch_size,
            utils.BRWAC_TXT_LEN)
        dict_dfs_events[name]['exp_name'] = name
        dict_dfs_events[name]['model_size'] = model_size

    return dict_dfs_events
