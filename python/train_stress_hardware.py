import functools
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

import t5
import numpy as np
from t5.data import preprocessors


import argparse
import json

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch_size', type=int, required=True, help='Batch size')

parser.add_argument('-n', '--name', type=str, required=True, help='GCS directory where models will be saved')
parser.add_argument('-t', '--txt_file', type=str, default='brwac_512.txt', help='Text file for training')
parser.add_argument('-ms', '--model_size', type=str, required=True, help='T5 model size')
parser.add_argument('-ns', '--nsteps', type=int, required=True, help='Number of steps to train')
parser.add_argument('-mp', '--model_parallelism', type=int, required=True, help='Model parallelism')
parser.add_argument('-s', '--seq_len', type=int, default=512, help='Sequence length for training')
parser.add_argument('-bp', '--pre_trained_dir', type=str, default='gs://t5-data/pretrained_models',help='GCS directory to load checkpoints from')

args = parser.parse_args()

args_json_dump = f'./argparse_dumps/{args.name}.json'
with open(args_json_dump, 'a') as f:
    print(f'Saving args to {args_json_dump} ...')
    json.dump(args.__dict__,f, indent=2)

# Traning data - same across all experiments
DATA_DIR = "gs://ptt5-1/data"
# Experiment specific parameters
BASE_DIR = f"gs://ptt5-1/{args.name}"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Connect to TPU
TPU_NAME = os.environ['TPU_NAME']
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
TPU_ADDRESS = tpu.get_master()


tf.disable_v2_behavior()

# Improve logging.
from contextlib import contextmanager
import logging as py_logging

tf.get_logger().propagate = False
py_logging.root.setLevel('INFO')

@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

# data preparation - datasets
TXT_FILE = os.path.join(DATA_DIR, args.txt_file)

def ptt5_dataset(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files
  del split

  # Load lines from the text file as examples.
  global TXT_FILE
  ds = tf.data.TextLineDataset(TXT_FILE)
  # Map each line to a {"sequence": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["targets"], ex)))
  return ds

def text_preprocessor(dataset):
  def _to_inputs_and_targets(input_dict):
    seq_str = input_dict['targets']
    return {"inputs": seq_str, "targets": seq_str}

  return dataset.map(_to_inputs_and_targets,num_parallel_calls=tf.data.experimental.AUTOTUNE)

def token_noise(dataset, output_features, **unused_kwargs):
  return preprocessors.denoise(dataset, output_features, 
                 noise_density=0.15, 
                 noise_mask_fn=preprocessors.iid_noise_mask, 
                 inputs_fn=preprocessors.noise_token_to_sentinel, 
                 targets_fn=None)

# Task 
t5.data.TaskRegistry.remove("denoise")
t5.data.TaskRegistry.add(
    "denoise",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=ptt5_dataset,
    splits=["train"],
    text_preprocessor=[text_preprocessor],
    token_preprocessor=[token_noise],
    metric_fns=[],
)
print("Added task.")

task = t5.data.TaskRegistry.get("denoise")
ds = task.get_dataset(split="train",sequence_length={"inputs": args.seq_len, "targets": args.seq_len})
print("A few preprocessed validation examples...")
for ex in tfds.as_numpy(ds.take(1)):
  inpt = ex["inputs"]
  tgt = ex["targets"]
  print((inpt == tgt).astype(np.float32).mean())

# Mixture
t5.data.MixtureRegistry.remove("train")
t5.data.MixtureRegistry.add(
    "train",
    ["denoise"],
     default_rate=1.0
)

# Model
MODEL_SIZE = args.model_size 
# Public GCS path for T5 pre-trained model checkpoints
BASE_PRETRAINED_DIR = args.pre_trained_dir
PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)

# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
#model_parallelism, train_batch_size, keep_checkpoint_max = {
#    "small": (1, 256//args.batch_div, 16),
#    "base": (2, 128//args.batch_div, 8),
#    "large": (8, 64//args.batch_div, 4),
#    "3B": (8, 16//args.batch_div, 1),
#    "11B": (8, 16//args.batch_div, 1)}[MODEL_SIZE]

tf.io.gfile.makedirs(MODEL_DIR)

model_parallelism = args.model_parallelism
batch_size = args.batch_size
keep_checkpoint_max = 1



# The models from our paper are based on the Mesh Tensorflow Transformer.
#
TPU_TOPOLOGY = "2x2"
model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": args.seq_len, "targets": args.seq_len},
    learning_rate_schedule=0.003,
    save_checkpoints_steps=5000,
   # keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
    keep_checkpoint_max=keep_checkpoint_max, 
    iterations_per_loop=100,
)

# "Pre-training" ==> FineTuning
def epoch_to_steps(batch_size,epochs, total_examples=7361359):
    return int((epochs * total_examples) // batch_size)

FINETUNE_STEPS =  epoch_to_steps(train_batch_size, args.nepoch)
FINETUNE_STEPS = args.nsteps
model.finetune(
    mixture_or_task_name="train",
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=FINETUNE_STEPS
)
