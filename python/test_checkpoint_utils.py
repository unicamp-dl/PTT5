from checkpoint_utils import *
import os

class simula_argparse(object):
    def __init__(self):
        self.batch_div = 1
        self.name = 'small_standard_vocab__TEMP__'
        self.txt_file = 'brwac_512.txt'
        self.model_size = "small"
        self.nepoch = 0.25
        self.seq_len = 512
        self.pre_trained_dir = 'gs://ptt5-1/small_standard_vocab__TEMP__/models'
        self.json_config_path = '../assin/T5_configs_json/ptt5-standard-vocab-small-config.json'
args = simula_argparse()

# Traning data - same across all experiments
DATA_DIR = "gs://ptt5-1/data"
# Experiment specific parameters
BASE_DIR = f"gs://ptt5-1/{args.name}"
MODELS_DIR = os.path.join(BASE_DIR, "models")
TXT_FILE = os.path.join(DATA_DIR, args.txt_file)

# Model
MODEL_SIZE = args.model_size
# Public GCS path for T5 pre-trained model checkpoints
BASE_PRETRAINED_DIR = args.pre_trained_dir
PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)

print('Testing local saving')
convert_tf_checkpoint_to_pytorch_gcs(
    tf.train.latest_checkpoint(MODEL_DIR),
    args.json_config_path,
    'ckpt-statedict-%d.pth' % get_checkpoint_step(tf.train.latest_checkpoint(MODEL_DIR))
    )

print('Testing GCS saving')
pytorch_dump_gcs = os.path.join(MODEL_DIR, 'checkpoints_pytorch',
                                f'pt-statedict-{args.name}-%d.pth' \
                 % get_checkpoint_step(tf.train.latest_checkpoint(MODEL_DIR)))
# print(pytorch_dump_gcs)


# Saving checkpoints and json config on cloud
# Caution: existing files will be overwritten!
tf.io.gfile.copy(
    args.json_config_path,
    os.path.join(MODEL_DIR,'checkpoints_pytorch'),
    overwrite=True
    )

convert_tf_checkpoint_to_pytorch_gcs(
    tf.train.latest_checkpoint(MODEL_DIR),
    args.json_config_path,
    pytorch_dump_gcs
    )
