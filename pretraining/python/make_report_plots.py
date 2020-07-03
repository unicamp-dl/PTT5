import sys
sys.path.append('../../utils/')
from utils import *

import tensorflow.compat.v1 as tf
import os
import pandas as pd
from t5.evaluation import eval_utils
import matplotlib.pyplot as plt
from collections import OrderedDict

dict_dfs_events = OrderedDict()
LOGS_DIR = '../logs_tensorboard'

# Experiments names - order is important to generate plots correctly
exp_names = [
    'small_standard_vocab', 'small_custom_sentencepiece_vocab',
    'base_standard_vocab', 'base_custom_sentencepiece_vocab',
    'large_standard_vocab', 'large_custom_sentencepiece_vocab'
]

# Parse tensorboard logs into pandas dataframes
# for name in sorted(os.listdir(LOGS_DIR), reverse=True):
for name in exp_names:
    dict_dfs_events[name] = tf_events_to_pandas(os.path.join(LOGS_DIR, name))
    dict_dfs_events[name]['fixed_step'] = fix_step_offset(
        dict_dfs_events[name]['step'])
    dict_dfs_events[name]['epoch'] = step_to_epoch(
        dict_dfs_events[name]['fixed_step'],
        DICT_BATCH_SIZE_PRETRAIN[get_model_size_from_dir(name)], BRWAC_TXT_LEN)

# where to look for events files - experiment names
labels_plots = {
    'small_standard_vocab': 'Small, T5 vocabulary',
    'base_standard_vocab': 'Base, T5 vocabulary',
    'large_standard_vocab': 'Large, T5 vocabulary',
    'small_custom_sentencepiece_vocab': 'Small, Portuguese vocabulary',
    'base_custom_sentencepiece_vocab': 'Base, Portuguese vocabulary',
    'large_custom_sentencepiece_vocab': 'Large, Portuguese vocabulary'
}

# Plot with all sizes
# fig = plt.figure()
fig, axs = plt.subplots(3, 2, sharex=False, sharey=False)

for n, (name, df) in enumerate(dict_dfs_events.items()):
    ax = axs.flat[n]
    ax.plot(df['epoch'], df['loss'], label=labels_plots[name])
    # plt.plot(df['step'], df['loss'], label=name)
    ax.legend()
    ax.grid()
    # ax.xlabel('Epoch')
    # ax.ylabel('Loss')
# plt.show()
# fig.tight_layout()
plt.savefig('../reports/pretraining_subplots.eps', dpi=1000, format='eps')

# Plot with all sizes
fig = plt.figure()
for name, df in dict_dfs_events.items():
    plt.plot(df['epoch'], df['loss'], label=labels_plots[name])
    # plt.plot(df['step'], df['loss'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('../reports/pretraining_all_sizes.eps', dpi=1000, format='eps')

# Plot with all sizes - with 0.9 smooth factor
fig = plt.figure()
for name, df in dict_dfs_events.items():
    plt.plot(df['epoch'],
             smooth_array(df['loss'], 0.9),
             label=labels_plots[name])
    # plt.plot(df['step'], df['loss'], label=name)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.savefig('../reports/pretraining_all_sizes_0.9_smooth.eps',
            dpi=1000,
            format='eps')
