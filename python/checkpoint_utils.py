import re
import os
import tempfile
import shutil
import tensorflow.compat.v1 as tf
from transformers.convert_t5_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
import torch

# Suppress some of the logging
import logging
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_t5").setLevel(logging.WARNING)

def get_checkpoint_step(ckpt_path):
    """Helper function to get step number for a given checkpoint
    Based on t5.models.mtf_model_get_latest_checkpoint_from_dir

    Args: 
        ckpt_path: str, checkpoint path

    Returns:
        int, checkpoint step number
    """
    return int(re.sub(".*ckpt-", "", ckpt_path))

def convert_tf_checkpoint_to_pytorch_gcs(tf_checkpoint_path, config_file,
                                         pytorch_dump_path):
    '''Wrapper for convert_tf_checkpoint_to_pytorch allowing checkpoints on GCS
    
    Args:
        tf_checkpoint_path: str, path for tensorflow checkpoint
        config_file: str, The config json file corresponding to the pre-trained T5 model.
        pytorch_dump_path: str,  Path to the output PyTorch model.
    '''
    
    files_to_download = tf.io.gfile.glob(tf_checkpoint_path + '.*')

    if not files_to_download:
        print(f'Checkpoint {tf_checkpoint_path} not found')
        return 
    
    with tempfile.TemporaryDirectory() as tmp_folder:
        for file in files_to_download:
            basename_file = os.path.basename(file)
            print(f"Downloading file {file} to {tmp_folder}")
            tf.io.gfile.copy(
                file,os.path.join(tmp_folder, basename_file),
                overwrite=True
            )
        
        ckpt_files = os.path.join(tmp_folder,
                                  os.path.basename(tf_checkpoint_path))
        convert_tf_checkpoint_to_pytorch(ckpt_files, config_file,
                                         pytorch_dump_path)

def compare_models(model_1, model_2):
    '''
    Based on: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    '''
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if (torch.equal(key_item_1[1], key_item_2[1])) \
        and (key_item_1[0], key_item_2[0]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

def compare_params_hface_models(base, comp):
    '''Compares two HuggingFace's  models

    Compare two HuggingFace models, checking for:
        - number of parameters
        - state_dict keys and values
    
    Args: 
        model1: HuggingFace model
        model2: HuggingFace model

    '''
    return_dict = {}
    # Number of parameters
    print('match_num_params_only_trainable: {}'.format(
    base.num_parameters(only_trainable=True) == \
    comp.num_parameters(only_trainable=True)))

    print('match_num_params_total: {}'.format(
    base.num_parameters(only_trainable=True) == \
    comp.num_parameters(only_trainable=True)))

    # state dict
    compare_models(base,comp)
    
    return  
