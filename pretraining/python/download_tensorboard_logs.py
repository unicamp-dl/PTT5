import tensorflow.compat.v1 as tf
import os
import tqdm

GCS_BUCKET = 'gs://ptt5-1'
TENSORBOARD_LOGS_LOCAL = '../logs_tensorboard'

os.makedirs(TENSORBOARD_LOGS_LOCAL, exist_ok=True)

# where to look for events files - experiment names
base_paths = [
        'small_standard_vocab',
        'base_standard_vocab',
        'large_standard_vocab',
        'small_custom_sentencepiece_vocab',
        'base_custom_sentencepiece_vocab',
        'large_custom_sentencepiece_vocab'
        ]


# all paths have the scructure
for base_path in base_paths:
    size = base_path.split('_')[0]
    full_path = os.path.join(GCS_BUCKET, base_path, 'models', size)
    download_dir = os.path.join(TENSORBOARD_LOGS_LOCAL, base_path)
    print(f'Downloading files from {full_path} to {download_dir}')
    os.makedirs(os.path.join(TENSORBOARD_LOGS_LOCAL,base_path), exist_ok=True)
    for file in tqdm.tqdm(tf.gfile.Glob(os.path.join(full_path, "events.*"))):
        tf.gfile.Copy(file,
                os.path.join(download_dir, os.path.basename(file)),
                overwrite=True
                )

