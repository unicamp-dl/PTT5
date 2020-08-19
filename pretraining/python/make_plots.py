import matplotlib.pyplot as plt
from parse_tensorboard_logs import parse_logs

labels_plots = {
    # Main initial experiments - all weights are updated
    'small_standard_vocab':
        'Small, T5 vocabulary',
    'base_standard_vocab':
        'Base, T5 vocabulary',
    'large_standard_vocab':
        'Large, T5 vocabulary',
    'small_custom_sentencepiece_vocab':
        'Small, Portuguese vocabulary',
    'base_custom_sentencepiece_vocab':
        'Base, Portuguese vocabulary',
    'large_custom_sentencepiece_vocab':
        'Large, Portuguese vocabulary',
    # Only embeddings are updated
    'small_embeddings_only_standard_vocab':
        'Small, T5 vocabulary',
    'small_embeddings_only_custom_sentencepiece_vocab':
        'Small, Portuguese vocabulary',
    'base_embeddings_only_standard_vocab':
        'Base, T5 vocabulary',
    'base_embeddings_only_custom_sentencepiece_vocab':
        'Base, Portuguese vocabulary',
    'large_embeddings_only_standard_vocab':
        'Large, T5 vocabulary',
    'large_embeddings_only_custom_sentencepiece_vocab':
        'Large, Portuguese vocabulary',
    # Double batch size for large (128 = 64 * 2)
    'large_batchsize_128_custom_sentencepiece_vocab':
        'Large, Portuguese vocabulary (128 batch size)',
    'large_batchsize_128_standard_vocab':
        'Large, T5 vocabulary (128 batch size)',
}


should_print = [
    # Main initial experiments - all weights are updated
    # 'small_standard_vocab',
    # 'small_custom_sentencepiece_vocab',
    'base_standard_vocab',
    'base_custom_sentencepiece_vocab',
    'large_standard_vocab',
    'large_custom_sentencepiece_vocab',
    # Only embeddings are updated
    # 'small_embeddings_only_standard_vocab',
    # 'small_embeddings_only_custom_sentencepiece_vocab',
    'base_embeddings_only_standard_vocab',
    'base_embeddings_only_custom_sentencepiece_vocab',
    'large_embeddings_only_standard_vocab',
    'large_embeddings_only_custom_sentencepiece_vocab',
    # Double batch size for large (128 = 64 * 2)
    # 'large_batchsize_128_custom_sentencepiece_vocab',
    # 'large_batchsize_128_standard_vocab',
]

dict_dfs_events = parse_logs()

plt.rcParams.update({'font.size': 16})

# All weights
eps_plot = 1e-3
fig = plt.figure()
for name, df in dict_dfs_events.items():
    if ('embedding' not in name) and (name in should_print):
        plt.plot(df['epoch'], df['loss'], label=labels_plots[name])
# plt.title('All weights')
plt.title('Whole model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0 - eps_plot, 4 + eps_plot)
plt.ylim(0, 0.3)
plt.legend(loc='best', fontsize='xx-small')
# plt.grid()
plt.tight_layout()
# plt.show()
plt.savefig('../plots/pretraing_all_weights.eps',
            dpi=1000,
            format='pdf',
            bbox_inches='tight')

# Embeddings only
fig = plt.figure()
for name, df in dict_dfs_events.items():
    if ('embedding' in name) and (name in should_print):
        plt.plot(df['epoch'], df['loss'], label=labels_plots[name])
plt.title('Only vocabulary embeddings')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0 - eps_plot, 4 + eps_plot)
plt.ylim(0, 0.3)
plt.legend(loc='best', fontsize='xx-small')
# plt.grid()
plt.tight_layout()
# plt.show()
plt.savefig('../plots/pretraing_embeddings_only.eps',
            dpi=1000,
            format='pdf',
            bbox_inches='tight')
