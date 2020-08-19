import sys
sys.path.append('../../utils/')
import utils
import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import OrderedDict, Tuple

LOGS_DIR = '../logs_tensorboard'

# Starting steps for each model size - from google GCS
INITIAL_STEP_MODEL_SIZE = {
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
            'step'] - INITIAL_STEP_MODEL_SIZE[model_size]
        dict_dfs_events[name]['epoch'] = utils.step_to_epoch(
            dict_dfs_events[name]['fixed_step'], batch_size,
            utils.BRWAC_TXT_LEN)
        dict_dfs_events[name]['exp_name'] = name
        dict_dfs_events[name]['model_size'] = model_size

    return dict_dfs_events


def analyze_epoch_time_num_params() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze time spent per epoch and number of parameters.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: First DataFrame contains data from
        each experiment, second is summarized by experiment group.
    """
    dict_dfs_events = parse_logs()
    # Keep only first epoch, assuming variation is minimal
    df_hours = pd.concat([
        (
            df[df['epoch'] <= 1.0][['exp_name', 'model_size', 'wall_time']]
            .groupby(['exp_name', 'model_size'])['wall_time']
            .agg(lambda x: x.max() - x.min())
            .reset_index()
        ) for df in dict_dfs_events.values()
        ])

    df_hours['wall_time_hours'] = df_hours['wall_time'] / 3600

    def group_hours_exp(x):
        """Maps experiment group from experiment name

        """
        label = utils.get_model_size_from_dir(x) + '-'
        # emb_only modifier
        if 'embedding' in x:
            label += 'embeddings-only'
        else:
            label += 'all-weights'
        # large with double batch size modifier
        if x.startswith('large_batchsize_128'):
            label += 'large-bs-128'
        return label

    df_hours['exp_group'] = df_hours['exp_name'].apply(group_hours_exp)
    df_hours = df_hours.sort_values(['exp_name', 'wall_time_hours'])

    # Parameter number - in case of embeddings only, make product explicit
    # Obs.: 770M vs 740M for large model
    map_param_number = {
        'base-all-weights': 222_903_552,
        'base-embeddings-only': 32128 * 768,
        'large-all-weights': 737_668_096,
        'large-all-weightslarge-bs-128': 737_668_096,
        'large-embeddings-only': 32128 * 1024,
        'small-all-weights': 60_506_624,
        'small-embeddings-only': 32128 * 512,
    }
    df_hours['param_num'] = df_hours['exp_group'].map(map_param_number)

    df_hours_group = (
        df_hours
        .groupby(['exp_group', 'param_num'])
        .agg(
        count=('wall_time_hours', 'size'),
        mean_hours_epoch=('wall_time_hours', 'mean'),
        std_hours_epoch=('wall_time_hours', 'std'))
        .reset_index()
        )

    df_hours_group['param_num_million'] = (df_hours_group['param_num'] /
                                           1e6).astype(int)
    df_hours_group['mean_hours_epoch_round'] = df_hours_group[
        'mean_hours_epoch'].round(1)

    df_hours_group['model_size'] = df_hours_group['exp_group'].apply(
        lambda x: x.split('-')[0].capitalize())

    df_hours_group['emb_only'] = df_hours_group['exp_group'].apply(
        lambda x: 'Yes' if 'embedd' in x else 'No')

    return df_hours, df_hours_group
