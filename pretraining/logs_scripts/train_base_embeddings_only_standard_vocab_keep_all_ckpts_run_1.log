Script started on 2020-07-26 19:52:08+00:00 [TERM="screen" TTY="/dev/pts/1" COLUMNS="152" LINES="52"]
marcospiau123@ptt5-vm5:~/PTT5/pretraining/bash$ g[Kls -lht
total 144K
-rw-r--r-- 1 marcospiau123 marcospiau123 276 Jul 26 19:52 train_base_embeddings_only_standard_vocab_keep_all_ckpts_run_1.sh
-rwxr-xr-x 1 marcospiau123 marcospiau123 298 Jul 26 19:51 [0m[01;32mtrain_base_custom_vocab_keep_all_ckpts_run_1.sh[0m
-rwxr-xr-x 1 marcospiau123 marcospiau123 332 Jul 26 19:51 [01;32mtrain_large_batchsize_128_custom_vocab_run_2.sh[0m
-rw-r--r-- 1 marcospiau123 marcospiau123 307 Jul 26 19:51 train_base_embeddings_only_custom_vocab_keep_all_ckpts_run_1.sh
-rw-r--r-- 1 marcospiau123 marcospiau123 202 Jul 26 19:51 train_base_standard_vocab_keep_all_ckpts_run_1.sh
-rwx------ 1 marcospiau123 marcospiau123 244 Jul 21 03:57 [01;32mtrain_large_batchsize_128_standard_vocab_run_2.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 259 Jul 21 03:52 [01;32mtrain_large_batchsize_128_custom_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 358 Jul 21 03:52 [01;32mtrain_large_embeddings_only_custom_vocab_run_3.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 183 Jul 17 07:08 [01;32mtrain_large_batchsize_128_standard_vocab_run_1.sh[0m
-rw-r--r-- 1 marcospiau123 marcospiau123 452 Jul 17 05:51 config_env.sh
-rw-r--r-- 1 marcospiau123 marcospiau123 113 Jul 17 05:51 create_tpu.sh
-rwxr-xr-x 1 marcospiau123 marcospiau123 157 Jul 17 05:51 [01;32mtest_small_continue_training.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 249 Jul 17 05:51 [01;32mtrain_base_custom_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 280 Jul 17 05:51 [01;32mtrain_base_embeddings_only_custom_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 203 Jul 17 05:51 [01;32mtrain_base_embeddings_only_stardard_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123  78 Jul 17 05:51 [01;32mtrain_base_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 205 Jul 17 05:51 [01;32mtrain_base_run_2.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 252 Jul 17 05:51 [01;32mtrain_large_custom_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 311 Jul 17 05:51 [01;32mtrain_large_custom_vocab_run_2.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 284 Jul 17 05:51 [01;32mtrain_large_embeddings_only_custom_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 358 Jul 17 05:51 [01;32mtrain_large_embeddings_only_custom_vocab_run_2.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 206 Jul 17 05:51 [01;32mtrain_large_embeddings_only_stardard_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 271 Jul 17 05:51 [01;32mtrain_large_embeddings_only_stardard_vocab_run_2.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 271 Jul 17 05:51 [01;32mtrain_large_embeddings_only_stardard_vocab_run_3.sh[0m
-rwx------ 1 marcospiau123 marcospiau123  80 Jul 17 05:51 [01;32mtrain_large_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 209 Jul 17 05:51 [01;32mtrain_large_run_2.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 212 Jul 17 05:51 [01;32mtrain_large_run_3.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 215 Jul 17 05:51 [01;32mtrain_large_run_4.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 212 Jul 17 05:51 [01;32mtrain_large_run_5.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 252 Jul 17 05:51 [01;32mtrain_small_custom_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 283 Jul 17 05:51 [01;32mtrain_small_embeddings_only_custom_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 206 Jul 17 05:51 [01;32mtrain_small_embeddings_only_stardard_vocab_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 271 Jul 17 05:51 [01;32mtrain_small_embeddings_only_stardard_vocab_run_2.sh[0m
-rwx------ 1 marcospiau123 marcospiau123  80 Jul 17 05:51 [01;32mtrain_small_run_1.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 209 Jul 17 05:51 [01;32mtrain_small_run_2.sh[0m
-rwx------ 1 marcospiau123 marcospiau123 143 Jul 17 05:51 [01;32mtrain_small_stress_hardware.sh[0m
marcospiau123@ptt5-vm5:~/PTT5/pretraining/bash$ cat train_base_embeddings_only_standard_vocab_keep_all_ckpts_run_1.sh
python3 ../python/train_v2.py \
-b 1 \
-ms base \
-e 1 \
-s 512 \
-jc '../../assin/T5_configs_json/ptt5-standard-vocab-base-config.json' \
--keep_all_checkpoints \
-n base_embeddings_only_standard_vocab_keep_all_ckpts  \
--train_embedding_only \
--save_checkpoints_steps 500

marcospiau123@ptt5-vm5:~/PTT5/pretraining/bash$ .[Kchmod 700 train_base_embeddings_only_standard_vocab_keep_all_ckpts_run_1.sh
marcospiau123@ptt5-vm5:~/PTT5/pretraining/bash$ ./train_base_embeddings_only_standard_vocab_keep_all_ckpts_run_1.sh
Arguments read from input: {'batch_div': 1.0, 'name': 'base_embeddings_only_standard_vocab_keep_all_ckpts', 'txt_file': 'brwac_512.txt', 'model_size': 'base', 'nepoch': 1.0, 'seq_len': 512, 'pre_trained_dir': 'gs://t5-data/pretrained_models', 'json_config_path': '../../assin/T5_configs_json/ptt5-standard-vocab-base-config.json', 'spiece_model_path': None, 'train_embedding_only': True, 'keep_all_checkpoints': True, 'save_checkpoints_steps': 500}
Saving args to ../argparse_dumps/base_embeddings_only_standard_vocab_keep_all_ckpts.json ...
INFO:googleapiclient.discovery:URL being requested: GET https://www.googleapis.com/discovery/v1/apis/tpu/v1/rest
INFO:googleapiclient.discovery:URL being requested: GET https://tpu.googleapis.com/v1/projects/ia376-1s2020-ptt5-2-282301/locations/europe-west4-a/nodes/ptt5-vm5?alt=json
INFO:oauth2client.transport:Attempting refresh to obtain initial access_token
WARNING:tensorflow:From /home/marcospiau123/.local/lib/python3.7/site-packages/tensorflow/python/compat/v2_compat.py:96: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
WARNING:tensorflow:From /home/marcospiau123/.local/lib/python3.7/site-packages/tensorflow/python/compat/v2_compat.py:96: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Added task.
2020-07-26 19:52:32.814567: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2020-07-26 19:52:32.814680: E tensorflow/stream_executor/cuda/cuda_driver.cc:313] failed call to cuInit: UNKNOWN ERROR (303)
2020-07-26 19:52:32.814853: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (ptt5-vm5): /proc/driver/nvidia/version does not exist
A few preprocessed validation examples...
2020-07-26 19:52:33.014759: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-26 19:52:33.025974: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2000179999 Hz
2020-07-26 19:52:33.026750: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f6f3c000b20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-26 19:52:33.026799: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
0.84375
INFO:tensorflow:Using config: {'_model_dir': 'gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': None, '_session_config': allow_soft_placement: true
cluster_def {
  job {
    name: "worker"
    tasks {
      key: 0
      value: "10.240.1.18:8470"
    }
  }
}
isolate_session_state: true
, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': None, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': ClusterSpec({'worker': ['10.240.1.18:8470']}), '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': 'grpc://10.240.1.18:8470', '_evaluation_master': 'grpc://10.240.1.18:8470', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1, '_tpu_config': TPUConfig(iterations_per_loop=100, num_shards=None, num_cores_per_replica=1, per_host_input_for_training=4, tpu_job_name=None, initial_infeed_sleep_secs=None, input_partition_dims=None, eval_training_input_configuration=2, experimental_host_call_every_n_steps=1), '_cluster': <tensorflow.python.distribute.cluster_resolver.tpu_cluster_resolver.TPUClusterResolver object at 0x7f6f585794e0>}
INFO:tensorflow:_TPUContext: eval_on_tpu True
INFO:tensorflow:Querying Tensorflow master (grpc://10.240.1.18:8470) for TPU system metadata.
2020-07-26 19:52:38.993454: W tensorflow/core/distributed_runtime/rpc/grpc_session.cc:373] GrpcSession::ListDevices will initialize the session with an empty graph and other defaults because the session has not yet been created.
INFO:tensorflow:Initializing TPU system (master: grpc://10.240.1.18:8470) to fetch topology for model parallelism. This might take a while.
INFO:tensorflow:Found TPU system:
INFO:tensorflow:*** Num TPU Cores: 8
INFO:tensorflow:*** Num TPU Workers: 1
INFO:tensorflow:*** Num TPU Cores Per Worker: 8
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, -1, -7316912904421034132)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 17179869184, -7444548491725204929)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 17179869184, -1756262523109711938)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 17179869184, -3821384974718020687)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 17179869184, -7715068273437434598)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 17179869184, 2715309689294849394)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 17179869184, -6564394846820060539)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 17179869184, -7894743291938435180)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 17179869184, 8898672848393713149)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 17179869184, -7388809041609880597)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 17179869184, -4873428955478824890)
WARNING:tensorflow:From /home/marcospiau123/.local/lib/python3.7/site-packages/tensorflow/python/ops/resource_variable_ops.py:1666: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /home/marcospiau123/.local/lib/python3.7/site-packages/tensorflow/python/training/training_util.py:236: Variable.initialized_value (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
Instructions for updating:
Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:enable_2d_tiling: False
INFO:tensorflow:num_cores_per_replica: 1
INFO:tensorflow:computation_shape: [1, 1, 1]
INFO:tensorflow:num_replicas: 8
INFO:tensorflow:device_assignment.topology.device_coordinates: [[[0 0 0]
  [0 0 1]
  [0 1 0]
  [0 1 1]
  [1 0 0]
  [1 0 1]
  [1 1 0]
  [1 1 1]]]
INFO:tensorflow:device_assignment.core_assignment: [[[0 0 0]]

 [[0 0 1]]

 [[0 1 0]]

 [[0 1 1]]

 [[1 0 0]]

 [[1 0 1]]

 [[1 1 0]]

 [[1 1 1]]]
WARNING:tensorflow:SimdMeshImpl ignoring devices ['', '', '', '', '', '', '', '']
INFO:tensorflow:SimdMeshImpl init: Shape[batch=4, model=2] LayoutRules{('vocab', 'model'), ('d_ff', 'model'), ('heads', 'model'), ('experts', 'batch'), ('ensemble', 'ensemble'), ('batch', 'batch')}
INFO:tensorflow:Device Assignment: <tensorflow.python.tpu.device_assignment.DeviceAssignment object at 0x7f6f586c6828>
INFO:tensorflow:serialize_num_microbatches: tokens_per_microbatch_per_replica=8192 batch_dim=Dimension(name='batch', size=128) sequence_length={'inputs': 512, 'targets': 512} batch_per_replica=32 num_microbatches=2
WARNING:tensorflow:Using default tf glorot_uniform_initializer for variable encoder/block_000/layer_000/SelfAttention/relative_attention_bias  The initialzer will guess the input and output dimensions  based on dimension order.
WARNING:tensorflow:Using default tf glorot_uniform_initializer for variable decoder/block_000/layer_000/SelfAttention/relative_attention_bias  The initialzer will guess the input and output dimensions  based on dimension order.
INFO:tensorflow:Variables being trained:
INFO:tensorflow:['shared/embedding']
INFO:tensorflow:Variables not being trained:
INFO:tensorflow:['encoder/block_000/layer_000/layer_norm/scale', 'encoder/block_000/layer_000/SelfAttention/q', 'encoder/block_000/layer_000/SelfAttention/k', 'encoder/block_000/layer_000/SelfAttention/v', 'encoder/block_000/layer_000/SelfAttention/o', 'encoder/block_000/layer_000/SelfAttention/relative_attention_bias', 'encoder/block_000/layer_001/layer_norm/scale', 'encoder/block_000/layer_001/DenseReluDense/wi/kernel', 'encoder/block_000/layer_001/DenseReluDense/wo/kernel', 'encoder/block_001/layer_000/layer_norm/scale', 'encoder/block_001/layer_000/SelfAttention/q', 'encoder/block_001/layer_000/SelfAttention/k', 'encoder/block_001/layer_000/SelfAttention/v', 'encoder/block_001/layer_000/SelfAttention/o', 'encoder/block_001/layer_001/layer_norm/scale', 'encoder/block_001/layer_001/DenseReluDense/wi/kernel', 'encoder/block_001/layer_001/DenseReluDense/wo/kernel', 'encoder/block_002/layer_000/layer_norm/scale', 'encoder/block_002/layer_000/SelfAttention/q', 'encoder/block_002/layer_000/SelfAttention/k', 'encoder/block_002/layer_000/SelfAttention/v', 'encoder/block_002/layer_000/SelfAttention/o', 'encoder/block_002/layer_001/layer_norm/scale', 'encoder/block_002/layer_001/DenseReluDense/wi/kernel', 'encoder/block_002/layer_001/DenseReluDense/wo/kernel', 'encoder/block_003/layer_000/layer_norm/scale', 'encoder/block_003/layer_000/SelfAttention/q', 'encoder/block_003/layer_000/SelfAttention/k', 'encoder/block_003/layer_000/SelfAttention/v', 'encoder/block_003/layer_000/SelfAttention/o', 'encoder/block_003/layer_001/layer_norm/scale', 'encoder/block_003/layer_001/DenseReluDense/wi/kernel', 'encoder/block_003/layer_001/DenseReluDense/wo/kernel', 'encoder/block_004/layer_000/layer_norm/scale', 'encoder/block_004/layer_000/SelfAttention/q', 'encoder/block_004/layer_000/SelfAttention/k', 'encoder/block_004/layer_000/SelfAttention/v', 'encoder/block_004/layer_000/SelfAttention/o', 'encoder/block_004/layer_001/layer_norm/scale', 'encoder/block_004/layer_001/DenseReluDense/wi/kernel', 'encoder/block_004/layer_001/DenseReluDense/wo/kernel', 'encoder/block_005/layer_000/layer_norm/scale', 'encoder/block_005/layer_000/SelfAttention/q', 'encoder/block_005/layer_000/SelfAttention/k', 'encoder/block_005/layer_000/SelfAttention/v', 'encoder/block_005/layer_000/SelfAttention/o', 'encoder/block_005/layer_001/layer_norm/scale', 'encoder/block_005/layer_001/DenseReluDense/wi/kernel', 'encoder/block_005/layer_001/DenseReluDense/wo/kernel', 'encoder/block_006/layer_000/layer_norm/scale', 'encoder/block_006/layer_000/SelfAttention/q', 'encoder/block_006/layer_000/SelfAttention/k', 'encoder/block_006/layer_000/SelfAttention/v', 'encoder/block_006/layer_000/SelfAttention/o', 'encoder/block_006/layer_001/layer_norm/scale', 'encoder/block_006/layer_001/DenseReluDense/wi/kernel', 'encoder/block_006/layer_001/DenseReluDense/wo/kernel', 'encoder/block_007/layer_000/layer_norm/scale', 'encoder/block_007/layer_000/SelfAttention/q', 'encoder/block_007/layer_000/SelfAttention/k', 'encoder/block_007/layer_000/SelfAttention/v', 'encoder/block_007/layer_000/SelfAttention/o', 'encoder/block_007/layer_001/layer_norm/scale', 'encoder/block_007/layer_001/DenseReluDense/wi/kernel', 'encoder/block_007/layer_001/DenseReluDense/wo/kernel', 'encoder/block_008/layer_000/layer_norm/scale', 'encoder/block_008/layer_000/SelfAttention/q', 'encoder/block_008/layer_000/SelfAttention/k', 'encoder/block_008/layer_000/SelfAttention/v', 'encoder/block_008/layer_000/SelfAttention/o', 'encoder/block_008/layer_001/layer_norm/scale', 'encoder/block_008/layer_001/DenseReluDense/wi/kernel', 'encoder/block_008/layer_001/DenseReluDense/wo/kernel', 'encoder/block_009/layer_000/layer_norm/scale', 'encoder/block_009/layer_000/SelfAttention/q', 'encoder/block_009/layer_000/SelfAttention/k', 'encoder/block_009/layer_000/SelfAttention/v', 'encoder/block_009/layer_000/SelfAttention/o', 'encoder/block_009/layer_001/layer_norm/scale', 'encoder/block_009/layer_001/DenseReluDense/wi/kernel', 'encoder/block_009/layer_001/DenseReluDense/wo/kernel', 'encoder/block_010/layer_000/layer_norm/scale', 'encoder/block_010/layer_000/SelfAttention/q', 'encoder/block_010/layer_000/SelfAttention/k', 'encoder/block_010/layer_000/SelfAttention/v', 'encoder/block_010/layer_000/SelfAttention/o', 'encoder/block_010/layer_001/layer_norm/scale', 'encoder/block_010/layer_001/DenseReluDense/wi/kernel', 'encoder/block_010/layer_001/DenseReluDense/wo/kernel', 'encoder/block_011/layer_000/layer_norm/scale', 'encoder/block_011/layer_000/SelfAttention/q', 'encoder/block_011/layer_000/SelfAttention/k', 'encoder/block_011/layer_000/SelfAttention/v', 'encoder/block_011/layer_000/SelfAttention/o', 'encoder/block_011/layer_001/layer_norm/scale', 'encoder/block_011/layer_001/DenseReluDense/wi/kernel', 'encoder/block_011/layer_001/DenseReluDense/wo/kernel', 'encoder/final_layer_norm/scale', 'decoder/block_000/layer_000/layer_norm/scale', 'decoder/block_000/layer_000/SelfAttention/q', 'decoder/block_000/layer_000/SelfAttention/k', 'decoder/block_000/layer_000/SelfAttention/v', 'decoder/block_000/layer_000/SelfAttention/o', 'decoder/block_000/layer_000/SelfAttention/relative_attention_bias', 'decoder/block_000/layer_001/layer_norm/scale', 'decoder/block_000/layer_001/EncDecAttention/q', 'decoder/block_000/layer_001/EncDecAttention/k', 'decoder/block_000/layer_001/EncDecAttention/v', 'decoder/block_000/layer_001/EncDecAttention/o', 'decoder/block_000/layer_002/layer_norm/scale', 'decoder/block_000/layer_002/DenseReluDense/wi/kernel', 'decoder/block_000/layer_002/DenseReluDense/wo/kernel', 'decoder/block_001/layer_000/layer_norm/scale', 'decoder/block_001/layer_000/SelfAttention/q', 'decoder/block_001/layer_000/SelfAttention/k', 'decoder/block_001/layer_000/SelfAttention/v', 'decoder/block_001/layer_000/SelfAttention/o', 'decoder/block_001/layer_001/layer_norm/scale', 'decoder/block_001/layer_001/EncDecAttention/q', 'decoder/block_001/layer_001/EncDecAttention/k', 'decoder/block_001/layer_001/EncDecAttention/v', 'decoder/block_001/layer_001/EncDecAttention/o', 'decoder/block_001/layer_002/layer_norm/scale', 'decoder/block_001/layer_002/DenseReluDense/wi/kernel', 'decoder/block_001/layer_002/DenseReluDense/wo/kernel', 'decoder/block_002/layer_000/layer_norm/scale', 'decoder/block_002/layer_000/SelfAttention/q', 'decoder/block_002/layer_000/SelfAttention/k', 'decoder/block_002/layer_000/SelfAttention/v', 'decoder/block_002/layer_000/SelfAttention/o', 'decoder/block_002/layer_001/layer_norm/scale', 'decoder/block_002/layer_001/EncDecAttention/q', 'decoder/block_002/layer_001/EncDecAttention/k', 'decoder/block_002/layer_001/EncDecAttention/v', 'decoder/block_002/layer_001/EncDecAttention/o', 'decoder/block_002/layer_002/layer_norm/scale', 'decoder/block_002/layer_002/DenseReluDense/wi/kernel', 'decoder/block_002/layer_002/DenseReluDense/wo/kernel', 'decoder/block_003/layer_000/layer_norm/scale', 'decoder/block_003/layer_000/SelfAttention/q', 'decoder/block_003/layer_000/SelfAttention/k', 'decoder/block_003/layer_000/SelfAttention/v', 'decoder/block_003/layer_000/SelfAttention/o', 'decoder/block_003/layer_001/layer_norm/scale', 'decoder/block_003/layer_001/EncDecAttention/q', 'decoder/block_003/layer_001/EncDecAttention/k', 'decoder/block_003/layer_001/EncDecAttention/v', 'decoder/block_003/layer_001/EncDecAttention/o', 'decoder/block_003/layer_002/layer_norm/scale', 'decoder/block_003/layer_002/DenseReluDense/wi/kernel', 'decoder/block_003/layer_002/DenseReluDense/wo/kernel', 'decoder/block_004/layer_000/layer_norm/scale', 'decoder/block_004/layer_000/SelfAttention/q', 'decoder/block_004/layer_000/SelfAttention/k', 'decoder/block_004/layer_000/SelfAttention/v', 'decoder/block_004/layer_000/SelfAttention/o', 'decoder/block_004/layer_001/layer_norm/scale', 'decoder/block_004/layer_001/EncDecAttention/q', 'decoder/block_004/layer_001/EncDecAttention/k', 'decoder/block_004/layer_001/EncDecAttention/v', 'decoder/block_004/layer_001/EncDecAttention/o', 'decoder/block_004/layer_002/layer_norm/scale', 'decoder/block_004/layer_002/DenseReluDense/wi/kernel', 'decoder/block_004/layer_002/DenseReluDense/wo/kernel', 'decoder/block_005/layer_000/layer_norm/scale', 'decoder/block_005/layer_000/SelfAttention/q', 'decoder/block_005/layer_000/SelfAttention/k', 'decoder/block_005/layer_000/SelfAttention/v', 'decoder/block_005/layer_000/SelfAttention/o', 'decoder/block_005/layer_001/layer_norm/scale', 'decoder/block_005/layer_001/EncDecAttention/q', 'decoder/block_005/layer_001/EncDecAttention/k', 'decoder/block_005/layer_001/EncDecAttention/v', 'decoder/block_005/layer_001/EncDecAttention/o', 'decoder/block_005/layer_002/layer_norm/scale', 'decoder/block_005/layer_002/DenseReluDense/wi/kernel', 'decoder/block_005/layer_002/DenseReluDense/wo/kernel', 'decoder/block_006/layer_000/layer_norm/scale', 'decoder/block_006/layer_000/SelfAttention/q', 'decoder/block_006/layer_000/SelfAttention/k', 'decoder/block_006/layer_000/SelfAttention/v', 'decoder/block_006/layer_000/SelfAttention/o', 'decoder/block_006/layer_001/layer_norm/scale', 'decoder/block_006/layer_001/EncDecAttention/q', 'decoder/block_006/layer_001/EncDecAttention/k', 'decoder/block_006/layer_001/EncDecAttention/v', 'decoder/block_006/layer_001/EncDecAttention/o', 'decoder/block_006/layer_002/layer_norm/scale', 'decoder/block_006/layer_002/DenseReluDense/wi/kernel', 'decoder/block_006/layer_002/DenseReluDense/wo/kernel', 'decoder/block_007/layer_000/layer_norm/scale', 'decoder/block_007/layer_000/SelfAttention/q', 'decoder/block_007/layer_000/SelfAttention/k', 'decoder/block_007/layer_000/SelfAttention/v', 'decoder/block_007/layer_000/SelfAttention/o', 'decoder/block_007/layer_001/layer_norm/scale', 'decoder/block_007/layer_001/EncDecAttention/q', 'decoder/block_007/layer_001/EncDecAttention/k', 'decoder/block_007/layer_001/EncDecAttention/v', 'decoder/block_007/layer_001/EncDecAttention/o', 'decoder/block_007/layer_002/layer_norm/scale', 'decoder/block_007/layer_002/DenseReluDense/wi/kernel', 'decoder/block_007/layer_002/DenseReluDense/wo/kernel', 'decoder/block_008/layer_000/layer_norm/scale', 'decoder/block_008/layer_000/SelfAttention/q', 'decoder/block_008/layer_000/SelfAttention/k', 'decoder/block_008/layer_000/SelfAttention/v', 'decoder/block_008/layer_000/SelfAttention/o', 'decoder/block_008/layer_001/layer_norm/scale', 'decoder/block_008/layer_001/EncDecAttention/q', 'decoder/block_008/layer_001/EncDecAttention/k', 'decoder/block_008/layer_001/EncDecAttention/v', 'decoder/block_008/layer_001/EncDecAttention/o', 'decoder/block_008/layer_002/layer_norm/scale', 'decoder/block_008/layer_002/DenseReluDense/wi/kernel', 'decoder/block_008/layer_002/DenseReluDense/wo/kernel', 'decoder/block_009/layer_000/layer_norm/scale', 'decoder/block_009/layer_000/SelfAttention/q', 'decoder/block_009/layer_000/SelfAttention/k', 'decoder/block_009/layer_000/SelfAttention/v', 'decoder/block_009/layer_000/SelfAttention/o', 'decoder/block_009/layer_001/layer_norm/scale', 'decoder/block_009/layer_001/EncDecAttention/q', 'decoder/block_009/layer_001/EncDecAttention/k', 'decoder/block_009/layer_001/EncDecAttention/v', 'decoder/block_009/layer_001/EncDecAttention/o', 'decoder/block_009/layer_002/layer_norm/scale', 'decoder/block_009/layer_002/DenseReluDense/wi/kernel', 'decoder/block_009/layer_002/DenseReluDense/wo/kernel', 'decoder/block_010/layer_000/layer_norm/scale', 'decoder/block_010/layer_000/SelfAttention/q', 'decoder/block_010/layer_000/SelfAttention/k', 'decoder/block_010/layer_000/SelfAttention/v', 'decoder/block_010/layer_000/SelfAttention/o', 'decoder/block_010/layer_001/layer_norm/scale', 'decoder/block_010/layer_001/EncDecAttention/q', 'decoder/block_010/layer_001/EncDecAttention/k', 'decoder/block_010/layer_001/EncDecAttention/v', 'decoder/block_010/layer_001/EncDecAttention/o', 'decoder/block_010/layer_002/layer_norm/scale', 'decoder/block_010/layer_002/DenseReluDense/wi/kernel', 'decoder/block_010/layer_002/DenseReluDense/wo/kernel', 'decoder/block_011/layer_000/layer_norm/scale', 'decoder/block_011/layer_000/SelfAttention/q', 'decoder/block_011/layer_000/SelfAttention/k', 'decoder/block_011/layer_000/SelfAttention/v', 'decoder/block_011/layer_000/SelfAttention/o', 'decoder/block_011/layer_001/layer_norm/scale', 'decoder/block_011/layer_001/EncDecAttention/q', 'decoder/block_011/layer_001/EncDecAttention/k', 'decoder/block_011/layer_001/EncDecAttention/v', 'decoder/block_011/layer_001/EncDecAttention/o', 'decoder/block_011/layer_002/layer_norm/scale', 'decoder/block_011/layer_002/DenseReluDense/wi/kernel', 'decoder/block_011/layer_002/DenseReluDense/wo/kernel', 'decoder/final_layer_norm/scale']
INFO:tensorflow:Create pnum_tensor
INFO:tensorflow:Casting <dtype: 'int32'> to float32 for allreduce
INFO:tensorflow:Variable decoder/block_000/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_000/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_000/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_000/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_000/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_000/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_000/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_000/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_000/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_000/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_001/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_001/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_001/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_001/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_001/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_001/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_001/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_001/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_001/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_001/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_002/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_002/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_002/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_002/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_002/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_002/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_002/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_002/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_002/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_002/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_003/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_003/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_003/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_003/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_003/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_003/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_003/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_003/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_003/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_003/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_004/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_004/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_004/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_004/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_004/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_004/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_004/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_004/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_004/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_004/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_005/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_005/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_005/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_005/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_005/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_005/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_005/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_005/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_005/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_005/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_006/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_006/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_006/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_006/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_006/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_006/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_006/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_006/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_006/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_006/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_007/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_007/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_007/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_007/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_007/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_007/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_007/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_007/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_007/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_007/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_008/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_008/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_008/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_008/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_008/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_008/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_008/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_008/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_008/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_008/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_009/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_009/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_009/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_009/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_009/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_009/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_009/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_009/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_009/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_009/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_010/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_010/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_010/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_010/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_010/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_010/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_010/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_010/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_010/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_010/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable decoder/block_011/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_011/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_011/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_011/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_011/layer_001/EncDecAttention/k                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_011/layer_001/EncDecAttention/o                size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable decoder/block_011/layer_001/EncDecAttention/q                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_011/layer_001/EncDecAttention/v                size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable decoder/block_011/layer_002/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable decoder/block_011/layer_002/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_000/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_000/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_000/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_000/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_000/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_000/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_001/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_001/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_001/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_001/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_001/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_001/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_002/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_002/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_002/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_002/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_002/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_002/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_003/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_003/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_003/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_003/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_003/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_003/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_004/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_004/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_004/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_004/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_004/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_004/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_005/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_005/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_005/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_005/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_005/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_005/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_006/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_006/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_006/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_006/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_006/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_006/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_007/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_007/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_007/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_007/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_007/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_007/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_008/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_008/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_008/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_008/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_008/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_008/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_009/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_009/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_009/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_009/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_009/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_009/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_010/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_010/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_010/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_010/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_010/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_010/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable encoder/block_011/layer_000/SelfAttention/k                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_011/layer_000/SelfAttention/o                  size 589824       slice_size 294912       Shape[heads=768, d_model=768]                               
INFO:tensorflow:Variable encoder/block_011/layer_000/SelfAttention/q                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_011/layer_000/SelfAttention/v                  size 589824       slice_size 294912       Shape[d_model=768, heads=768]                               
INFO:tensorflow:Variable encoder/block_011/layer_001/DenseReluDense/wi/kernel         size 2359296      slice_size 1179648      Shape[d_model=768, d_ff=3072]                               
INFO:tensorflow:Variable encoder/block_011/layer_001/DenseReluDense/wo/kernel         size 2359296      slice_size 1179648      Shape[d_ff=3072, d_model=768]                               
INFO:tensorflow:Variable shared/embedding                                             size 24674304     slice_size 12337152     Shape[vocab=32128, d_model=768]                             
INFO:tensorflow:Variable stacked/encoder/block_000/layer_000/SelfAttention/relative_attention_bias size 768          slice_size 384          Shape[stacked=2, heads=12, buckets=32]                      
INFO:tensorflow:    encoder/block_000/layer_000/SelfAttention/relative_attention_bias
INFO:tensorflow:    decoder/block_000/layer_000/SelfAttention/relative_attention_bias
INFO:tensorflow:Variable stacked/encoder/block_000/layer_000/layer_norm/scale         size 47616        slice_size 47616        Shape[stacked=62, d_model=768]                              
INFO:tensorflow:    encoder/block_000/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_000/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_001/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_001/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_002/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_002/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_003/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_003/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_004/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_004/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_005/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_005/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_006/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_006/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_007/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_007/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_008/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_008/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_009/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_009/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_010/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_010/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/block_011/layer_000/layer_norm/scale
INFO:tensorflow:    encoder/block_011/layer_001/layer_norm/scale
INFO:tensorflow:    encoder/final_layer_norm/scale
INFO:tensorflow:    decoder/block_000/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_000/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_000/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_001/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_001/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_001/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_002/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_002/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_002/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_003/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_003/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_003/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_004/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_004/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_004/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_005/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_005/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_005/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_006/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_006/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_006/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_007/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_007/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_007/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_008/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_008/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_008/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_009/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_009/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_009/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_010/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_010/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_010/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/block_011/layer_000/layer_norm/scale
INFO:tensorflow:    decoder/block_011/layer_001/layer_norm/scale
INFO:tensorflow:    decoder/block_011/layer_002/layer_norm/scale
INFO:tensorflow:    decoder/final_layer_norm/scale
INFO:tensorflow:Trainable Variables            count: 195     Total size: 222903552        Total slice_size: 111475584      
INFO:tensorflow:All Variables                  count: 197     Total size: 222936448        Total slice_size: 111492416      
INFO:tensorflow:Counters:
allconcat: 5.24e+05
 allconcat/0: 5.24e+05
  allconcat/0/reshape_op: 5.24e+05
allreduce: 6.64e+09
 allreduce/[0]: 4.01e+08
  allreduce/[0]/einsum_op: 9.87e+07
  allreduce/[0]/reduce_op: 3.02e+08
 allreduce/[1]: 6.24e+09
  allreduce/[1]/einsum_op: 6.24e+09
  allreduce/[1]/reduce_op: 2.68e+05
einsum: 2.97e+13
einsum_unique: 2.96e+13
output: 2.78e+09
 output/AddOperation: 1.35e+05
 output/Constant: 8
 output/EinsumOperation: 3.95e+08
 output/ImportOperation: 3.15e+06
 output/MinMaxOperation: 16
 output/ReduceOperation: 1.35e+05
 output/ReshapeOperation: 1.31e+06
 output/ScalarAddOperation: 9.87e+07
 output/ScalarMultiplyOperation: 4.04e+05
 output/SlicewiseOperation: 4.94e+08
 output/StackedVariable: 3.84e+05
 output/UnstackOperation: 3.84e+05
 output/Variable: 8.92e+08
 output/WhileLoopOperation: 8.92e+08
output_unique: 6.94e+08
 output_unique/AddOperation: 3.29e+04
 output_unique/Constant: 1
 output_unique/EinsumOperation: 9.87e+07
 output_unique/ImportOperation: 3.93e+05
 output_unique/MinMaxOperation: 2
 output_unique/ReduceOperation: 3.29e+04
 output_unique/ReshapeOperation: 4.59e+05
 output_unique/ScalarAddOperation: 2.47e+07
 output_unique/ScalarMultiplyOperation: 9.87e+04
 output_unique/SlicewiseOperation: 1.23e+08
 output_unique/StackedVariable: 4.84e+04
 output_unique/UnstackOperation: 4.84e+04
 output_unique/Variable: 2.23e+08
 output_unique/WhileLoopOperation: 2.23e+08
variables: 2.23e+08
 variables/trainable: 2.23e+08
 variables/untrainable: 3.29e+04
INFO:tensorflow:Initializing variables from gs://t5-data/pretrained_models/base/model.ckpt-999900:
INFO:tensorflow:Variables in gs://t5-data/pretrained_models/base/model.ckpt-999900 but not in graph:
INFO:tensorflow:decoder/block_000/layer_000/SelfAttention/k_slot_vc
decoder/block_000/layer_000/SelfAttention/k_slot_vr
decoder/block_000/layer_000/SelfAttention/o_slot_vc
decoder/block_000/layer_000/SelfAttention/o_slot_vr
decoder/block_000/layer_000/SelfAttention/q_slot_vc
decoder/block_000/layer_000/SelfAttention/q_slot_vr
decoder/block_000/layer_000/SelfAttention/relative_attention_bias_slot_v
decoder/block_000/layer_000/SelfAttention/v_slot_vc
decoder/block_000/layer_000/SelfAttention/v_slot_vr
decoder/block_000/layer_000/layer_norm/scale_slot_v
decoder/block_000/layer_001/EncDecAttention/k_slot_vc
decoder/block_000/layer_001/EncDecAttention/k_slot_vr
decoder/block_000/layer_001/EncDecAttention/o_slot_vc
decoder/block_000/layer_001/EncDecAttention/o_slot_vr
decoder/block_000/layer_001/EncDecAttention/q_slot_vc
decoder/block_000/layer_001/EncDecAttention/q_slot_vr
decoder/block_000/layer_001/EncDecAttention/v_slot_vc
decoder/block_000/layer_001/EncDecAttention/v_slot_vr
decoder/block_000/layer_001/layer_norm/scale_slot_v
decoder/block_000/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_000/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_000/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_000/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_000/layer_002/layer_norm/scale_slot_v
decoder/block_001/layer_000/SelfAttention/k_slot_vc
decoder/block_001/layer_000/SelfAttention/k_slot_vr
decoder/block_001/layer_000/SelfAttention/o_slot_vc
decoder/block_001/layer_000/SelfAttention/o_slot_vr
decoder/block_001/layer_000/SelfAttention/q_slot_vc
decoder/block_001/layer_000/SelfAttention/q_slot_vr
decoder/block_001/layer_000/SelfAttention/v_slot_vc
decoder/block_001/layer_000/SelfAttention/v_slot_vr
decoder/block_001/layer_000/layer_norm/scale_slot_v
decoder/block_001/layer_001/EncDecAttention/k_slot_vc
decoder/block_001/layer_001/EncDecAttention/k_slot_vr
decoder/block_001/layer_001/EncDecAttention/o_slot_vc
decoder/block_001/layer_001/EncDecAttention/o_slot_vr
decoder/block_001/layer_001/EncDecAttention/q_slot_vc
decoder/block_001/layer_001/EncDecAttention/q_slot_vr
decoder/block_001/layer_001/EncDecAttention/v_slot_vc
decoder/block_001/layer_001/EncDecAttention/v_slot_vr
decoder/block_001/layer_001/layer_norm/scale_slot_v
decoder/block_001/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_001/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_001/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_001/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_001/layer_002/layer_norm/scale_slot_v
decoder/block_002/layer_000/SelfAttention/k_slot_vc
decoder/block_002/layer_000/SelfAttention/k_slot_vr
decoder/block_002/layer_000/SelfAttention/o_slot_vc
decoder/block_002/layer_000/SelfAttention/o_slot_vr
decoder/block_002/layer_000/SelfAttention/q_slot_vc
decoder/block_002/layer_000/SelfAttention/q_slot_vr
decoder/block_002/layer_000/SelfAttention/v_slot_vc
decoder/block_002/layer_000/SelfAttention/v_slot_vr
decoder/block_002/layer_000/layer_norm/scale_slot_v
decoder/block_002/layer_001/EncDecAttention/k_slot_vc
decoder/block_002/layer_001/EncDecAttention/k_slot_vr
decoder/block_002/layer_001/EncDecAttention/o_slot_vc
decoder/block_002/layer_001/EncDecAttention/o_slot_vr
decoder/block_002/layer_001/EncDecAttention/q_slot_vc
decoder/block_002/layer_001/EncDecAttention/q_slot_vr
decoder/block_002/layer_001/EncDecAttention/v_slot_vc
decoder/block_002/layer_001/EncDecAttention/v_slot_vr
decoder/block_002/layer_001/layer_norm/scale_slot_v
decoder/block_002/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_002/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_002/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_002/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_002/layer_002/layer_norm/scale_slot_v
decoder/block_003/layer_000/SelfAttention/k_slot_vc
decoder/block_003/layer_000/SelfAttention/k_slot_vr
decoder/block_003/layer_000/SelfAttention/o_slot_vc
decoder/block_003/layer_000/SelfAttention/o_slot_vr
decoder/block_003/layer_000/SelfAttention/q_slot_vc
decoder/block_003/layer_000/SelfAttention/q_slot_vr
decoder/block_003/layer_000/SelfAttention/v_slot_vc
decoder/block_003/layer_000/SelfAttention/v_slot_vr
decoder/block_003/layer_000/layer_norm/scale_slot_v
decoder/block_003/layer_001/EncDecAttention/k_slot_vc
decoder/block_003/layer_001/EncDecAttention/k_slot_vr
decoder/block_003/layer_001/EncDecAttention/o_slot_vc
decoder/block_003/layer_001/EncDecAttention/o_slot_vr
decoder/block_003/layer_001/EncDecAttention/q_slot_vc
decoder/block_003/layer_001/EncDecAttention/q_slot_vr
decoder/block_003/layer_001/EncDecAttention/v_slot_vc
decoder/block_003/layer_001/EncDecAttention/v_slot_vr
decoder/block_003/layer_001/layer_norm/scale_slot_v
decoder/block_003/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_003/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_003/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_003/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_003/layer_002/layer_norm/scale_slot_v
decoder/block_004/layer_000/SelfAttention/k_slot_vc
decoder/block_004/layer_000/SelfAttention/k_slot_vr
decoder/block_004/layer_000/SelfAttention/o_slot_vc
decoder/block_004/layer_000/SelfAttention/o_slot_vr
decoder/block_004/layer_000/SelfAttention/q_slot_vc
decoder/block_004/layer_000/SelfAttention/q_slot_vr
decoder/block_004/layer_000/SelfAttention/v_slot_vc
decoder/block_004/layer_000/SelfAttention/v_slot_vr
decoder/block_004/layer_000/layer_norm/scale_slot_v
decoder/block_004/layer_001/EncDecAttention/k_slot_vc
decoder/block_004/layer_001/EncDecAttention/k_slot_vr
decoder/block_004/layer_001/EncDecAttention/o_slot_vc
decoder/block_004/layer_001/EncDecAttention/o_slot_vr
decoder/block_004/layer_001/EncDecAttention/q_slot_vc
decoder/block_004/layer_001/EncDecAttention/q_slot_vr
decoder/block_004/layer_001/EncDecAttention/v_slot_vc
decoder/block_004/layer_001/EncDecAttention/v_slot_vr
decoder/block_004/layer_001/layer_norm/scale_slot_v
decoder/block_004/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_004/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_004/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_004/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_004/layer_002/layer_norm/scale_slot_v
decoder/block_005/layer_000/SelfAttention/k_slot_vc
decoder/block_005/layer_000/SelfAttention/k_slot_vr
decoder/block_005/layer_000/SelfAttention/o_slot_vc
decoder/block_005/layer_000/SelfAttention/o_slot_vr
decoder/block_005/layer_000/SelfAttention/q_slot_vc
decoder/block_005/layer_000/SelfAttention/q_slot_vr
decoder/block_005/layer_000/SelfAttention/v_slot_vc
decoder/block_005/layer_000/SelfAttention/v_slot_vr
decoder/block_005/layer_000/layer_norm/scale_slot_v
decoder/block_005/layer_001/EncDecAttention/k_slot_vc
decoder/block_005/layer_001/EncDecAttention/k_slot_vr
decoder/block_005/layer_001/EncDecAttention/o_slot_vc
decoder/block_005/layer_001/EncDecAttention/o_slot_vr
decoder/block_005/layer_001/EncDecAttention/q_slot_vc
decoder/block_005/layer_001/EncDecAttention/q_slot_vr
decoder/block_005/layer_001/EncDecAttention/v_slot_vc
decoder/block_005/layer_001/EncDecAttention/v_slot_vr
decoder/block_005/layer_001/layer_norm/scale_slot_v
decoder/block_005/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_005/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_005/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_005/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_005/layer_002/layer_norm/scale_slot_v
decoder/block_006/layer_000/SelfAttention/k_slot_vc
decoder/block_006/layer_000/SelfAttention/k_slot_vr
decoder/block_006/layer_000/SelfAttention/o_slot_vc
decoder/block_006/layer_000/SelfAttention/o_slot_vr
decoder/block_006/layer_000/SelfAttention/q_slot_vc
decoder/block_006/layer_000/SelfAttention/q_slot_vr
decoder/block_006/layer_000/SelfAttention/v_slot_vc
decoder/block_006/layer_000/SelfAttention/v_slot_vr
decoder/block_006/layer_000/layer_norm/scale_slot_v
decoder/block_006/layer_001/EncDecAttention/k_slot_vc
decoder/block_006/layer_001/EncDecAttention/k_slot_vr
decoder/block_006/layer_001/EncDecAttention/o_slot_vc
decoder/block_006/layer_001/EncDecAttention/o_slot_vr
decoder/block_006/layer_001/EncDecAttention/q_slot_vc
decoder/block_006/layer_001/EncDecAttention/q_slot_vr
decoder/block_006/layer_001/EncDecAttention/v_slot_vc
decoder/block_006/layer_001/EncDecAttention/v_slot_vr
decoder/block_006/layer_001/layer_norm/scale_slot_v
decoder/block_006/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_006/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_006/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_006/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_006/layer_002/layer_norm/scale_slot_v
decoder/block_007/layer_000/SelfAttention/k_slot_vc
decoder/block_007/layer_000/SelfAttention/k_slot_vr
decoder/block_007/layer_000/SelfAttention/o_slot_vc
decoder/block_007/layer_000/SelfAttention/o_slot_vr
decoder/block_007/layer_000/SelfAttention/q_slot_vc
decoder/block_007/layer_000/SelfAttention/q_slot_vr
decoder/block_007/layer_000/SelfAttention/v_slot_vc
decoder/block_007/layer_000/SelfAttention/v_slot_vr
decoder/block_007/layer_000/layer_norm/scale_slot_v
decoder/block_007/layer_001/EncDecAttention/k_slot_vc
decoder/block_007/layer_001/EncDecAttention/k_slot_vr
decoder/block_007/layer_001/EncDecAttention/o_slot_vc
decoder/block_007/layer_001/EncDecAttention/o_slot_vr
decoder/block_007/layer_001/EncDecAttention/q_slot_vc
decoder/block_007/layer_001/EncDecAttention/q_slot_vr
decoder/block_007/layer_001/EncDecAttention/v_slot_vc
decoder/block_007/layer_001/EncDecAttention/v_slot_vr
decoder/block_007/layer_001/layer_norm/scale_slot_v
decoder/block_007/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_007/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_007/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_007/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_007/layer_002/layer_norm/scale_slot_v
decoder/block_008/layer_000/SelfAttention/k_slot_vc
decoder/block_008/layer_000/SelfAttention/k_slot_vr
decoder/block_008/layer_000/SelfAttention/o_slot_vc
decoder/block_008/layer_000/SelfAttention/o_slot_vr
decoder/block_008/layer_000/SelfAttention/q_slot_vc
decoder/block_008/layer_000/SelfAttention/q_slot_vr
decoder/block_008/layer_000/SelfAttention/v_slot_vc
decoder/block_008/layer_000/SelfAttention/v_slot_vr
decoder/block_008/layer_000/layer_norm/scale_slot_v
decoder/block_008/layer_001/EncDecAttention/k_slot_vc
decoder/block_008/layer_001/EncDecAttention/k_slot_vr
decoder/block_008/layer_001/EncDecAttention/o_slot_vc
decoder/block_008/layer_001/EncDecAttention/o_slot_vr
decoder/block_008/layer_001/EncDecAttention/q_slot_vc
decoder/block_008/layer_001/EncDecAttention/q_slot_vr
decoder/block_008/layer_001/EncDecAttention/v_slot_vc
decoder/block_008/layer_001/EncDecAttention/v_slot_vr
decoder/block_008/layer_001/layer_norm/scale_slot_v
decoder/block_008/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_008/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_008/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_008/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_008/layer_002/layer_norm/scale_slot_v
decoder/block_009/layer_000/SelfAttention/k_slot_vc
decoder/block_009/layer_000/SelfAttention/k_slot_vr
decoder/block_009/layer_000/SelfAttention/o_slot_vc
decoder/block_009/layer_000/SelfAttention/o_slot_vr
decoder/block_009/layer_000/SelfAttention/q_slot_vc
decoder/block_009/layer_000/SelfAttention/q_slot_vr
decoder/block_009/layer_000/SelfAttention/v_slot_vc
decoder/block_009/layer_000/SelfAttention/v_slot_vr
decoder/block_009/layer_000/layer_norm/scale_slot_v
decoder/block_009/layer_001/EncDecAttention/k_slot_vc
decoder/block_009/layer_001/EncDecAttention/k_slot_vr
decoder/block_009/layer_001/EncDecAttention/o_slot_vc
decoder/block_009/layer_001/EncDecAttention/o_slot_vr
decoder/block_009/layer_001/EncDecAttention/q_slot_vc
decoder/block_009/layer_001/EncDecAttention/q_slot_vr
decoder/block_009/layer_001/EncDecAttention/v_slot_vc
decoder/block_009/layer_001/EncDecAttention/v_slot_vr
decoder/block_009/layer_001/layer_norm/scale_slot_v
decoder/block_009/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_009/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_009/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_009/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_009/layer_002/layer_norm/scale_slot_v
decoder/block_010/layer_000/SelfAttention/k_slot_vc
decoder/block_010/layer_000/SelfAttention/k_slot_vr
decoder/block_010/layer_000/SelfAttention/o_slot_vc
decoder/block_010/layer_000/SelfAttention/o_slot_vr
decoder/block_010/layer_000/SelfAttention/q_slot_vc
decoder/block_010/layer_000/SelfAttention/q_slot_vr
decoder/block_010/layer_000/SelfAttention/v_slot_vc
decoder/block_010/layer_000/SelfAttention/v_slot_vr
decoder/block_010/layer_000/layer_norm/scale_slot_v
decoder/block_010/layer_001/EncDecAttention/k_slot_vc
decoder/block_010/layer_001/EncDecAttention/k_slot_vr
decoder/block_010/layer_001/EncDecAttention/o_slot_vc
decoder/block_010/layer_001/EncDecAttention/o_slot_vr
decoder/block_010/layer_001/EncDecAttention/q_slot_vc
decoder/block_010/layer_001/EncDecAttention/q_slot_vr
decoder/block_010/layer_001/EncDecAttention/v_slot_vc
decoder/block_010/layer_001/EncDecAttention/v_slot_vr
decoder/block_010/layer_001/layer_norm/scale_slot_v
decoder/block_010/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_010/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_010/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_010/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_010/layer_002/layer_norm/scale_slot_v
decoder/block_011/layer_000/SelfAttention/k_slot_vc
decoder/block_011/layer_000/SelfAttention/k_slot_vr
decoder/block_011/layer_000/SelfAttention/o_slot_vc
decoder/block_011/layer_000/SelfAttention/o_slot_vr
decoder/block_011/layer_000/SelfAttention/q_slot_vc
decoder/block_011/layer_000/SelfAttention/q_slot_vr
decoder/block_011/layer_000/SelfAttention/v_slot_vc
decoder/block_011/layer_000/SelfAttention/v_slot_vr
decoder/block_011/layer_000/layer_norm/scale_slot_v
decoder/block_011/layer_001/EncDecAttention/k_slot_vc
decoder/block_011/layer_001/EncDecAttention/k_slot_vr
decoder/block_011/layer_001/EncDecAttention/o_slot_vc
decoder/block_011/layer_001/EncDecAttention/o_slot_vr
decoder/block_011/layer_001/EncDecAttention/q_slot_vc
decoder/block_011/layer_001/EncDecAttention/q_slot_vr
decoder/block_011/layer_001/EncDecAttention/v_slot_vc
decoder/block_011/layer_001/EncDecAttention/v_slot_vr
decoder/block_011/layer_001/layer_norm/scale_slot_v
decoder/block_011/layer_002/DenseReluDense/wi/kernel_slot_vc
decoder/block_011/layer_002/DenseReluDense/wi/kernel_slot_vr
decoder/block_011/layer_002/DenseReluDense/wo/kernel_slot_vc
decoder/block_011/layer_002/DenseReluDense/wo/kernel_slot_vr
decoder/block_011/layer_002/layer_norm/scale_slot_v
decoder/final_layer_norm/scale_slot_v
encoder/block_000/layer_000/SelfAttention/k_slot_vc
encoder/block_000/layer_000/SelfAttention/k_slot_vr
encoder/block_000/layer_000/SelfAttention/o_slot_vc
encoder/block_000/layer_000/SelfAttention/o_slot_vr
encoder/block_000/layer_000/SelfAttention/q_slot_vc
encoder/block_000/layer_000/SelfAttention/q_slot_vr
encoder/block_000/layer_000/SelfAttention/relative_attention_bias_slot_v
encoder/block_000/layer_000/SelfAttention/v_slot_vc
encoder/block_000/layer_000/SelfAttention/v_slot_vr
encoder/block_000/layer_000/layer_norm/scale_slot_v
encoder/block_000/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_000/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_000/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_000/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_000/layer_001/layer_norm/scale_slot_v
encoder/block_001/layer_000/SelfAttention/k_slot_vc
encoder/block_001/layer_000/SelfAttention/k_slot_vr
encoder/block_001/layer_000/SelfAttention/o_slot_vc
encoder/block_001/layer_000/SelfAttention/o_slot_vr
encoder/block_001/layer_000/SelfAttention/q_slot_vc
encoder/block_001/layer_000/SelfAttention/q_slot_vr
encoder/block_001/layer_000/SelfAttention/v_slot_vc
encoder/block_001/layer_000/SelfAttention/v_slot_vr
encoder/block_001/layer_000/layer_norm/scale_slot_v
encoder/block_001/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_001/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_001/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_001/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_001/layer_001/layer_norm/scale_slot_v
encoder/block_002/layer_000/SelfAttention/k_slot_vc
encoder/block_002/layer_000/SelfAttention/k_slot_vr
encoder/block_002/layer_000/SelfAttention/o_slot_vc
encoder/block_002/layer_000/SelfAttention/o_slot_vr
encoder/block_002/layer_000/SelfAttention/q_slot_vc
encoder/block_002/layer_000/SelfAttention/q_slot_vr
encoder/block_002/layer_000/SelfAttention/v_slot_vc
encoder/block_002/layer_000/SelfAttention/v_slot_vr
encoder/block_002/layer_000/layer_norm/scale_slot_v
encoder/block_002/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_002/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_002/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_002/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_002/layer_001/layer_norm/scale_slot_v
encoder/block_003/layer_000/SelfAttention/k_slot_vc
encoder/block_003/layer_000/SelfAttention/k_slot_vr
encoder/block_003/layer_000/SelfAttention/o_slot_vc
encoder/block_003/layer_000/SelfAttention/o_slot_vr
encoder/block_003/layer_000/SelfAttention/q_slot_vc
encoder/block_003/layer_000/SelfAttention/q_slot_vr
encoder/block_003/layer_000/SelfAttention/v_slot_vc
encoder/block_003/layer_000/SelfAttention/v_slot_vr
encoder/block_003/layer_000/layer_norm/scale_slot_v
encoder/block_003/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_003/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_003/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_003/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_003/layer_001/layer_norm/scale_slot_v
encoder/block_004/layer_000/SelfAttention/k_slot_vc
encoder/block_004/layer_000/SelfAttention/k_slot_vr
encoder/block_004/layer_000/SelfAttention/o_slot_vc
encoder/block_004/layer_000/SelfAttention/o_slot_vr
encoder/block_004/layer_000/SelfAttention/q_slot_vc
encoder/block_004/layer_000/SelfAttention/q_slot_vr
encoder/block_004/layer_000/SelfAttention/v_slot_vc
encoder/block_004/layer_000/SelfAttention/v_slot_vr
encoder/block_004/layer_000/layer_norm/scale_slot_v
encoder/block_004/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_004/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_004/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_004/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_004/layer_001/layer_norm/scale_slot_v
encoder/block_005/layer_000/SelfAttention/k_slot_vc
encoder/block_005/layer_000/SelfAttention/k_slot_vr
encoder/block_005/layer_000/SelfAttention/o_slot_vc
encoder/block_005/layer_000/SelfAttention/o_slot_vr
encoder/block_005/layer_000/SelfAttention/q_slot_vc
encoder/block_005/layer_000/SelfAttention/q_slot_vr
encoder/block_005/layer_000/SelfAttention/v_slot_vc
encoder/block_005/layer_000/SelfAttention/v_slot_vr
encoder/block_005/layer_000/layer_norm/scale_slot_v
encoder/block_005/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_005/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_005/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_005/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_005/layer_001/layer_norm/scale_slot_v
encoder/block_006/layer_000/SelfAttention/k_slot_vc
encoder/block_006/layer_000/SelfAttention/k_slot_vr
encoder/block_006/layer_000/SelfAttention/o_slot_vc
encoder/block_006/layer_000/SelfAttention/o_slot_vr
encoder/block_006/layer_000/SelfAttention/q_slot_vc
encoder/block_006/layer_000/SelfAttention/q_slot_vr
encoder/block_006/layer_000/SelfAttention/v_slot_vc
encoder/block_006/layer_000/SelfAttention/v_slot_vr
encoder/block_006/layer_000/layer_norm/scale_slot_v
encoder/block_006/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_006/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_006/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_006/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_006/layer_001/layer_norm/scale_slot_v
encoder/block_007/layer_000/SelfAttention/k_slot_vc
encoder/block_007/layer_000/SelfAttention/k_slot_vr
encoder/block_007/layer_000/SelfAttention/o_slot_vc
encoder/block_007/layer_000/SelfAttention/o_slot_vr
encoder/block_007/layer_000/SelfAttention/q_slot_vc
encoder/block_007/layer_000/SelfAttention/q_slot_vr
encoder/block_007/layer_000/SelfAttention/v_slot_vc
encoder/block_007/layer_000/SelfAttention/v_slot_vr
encoder/block_007/layer_000/layer_norm/scale_slot_v
encoder/block_007/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_007/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_007/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_007/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_007/layer_001/layer_norm/scale_slot_v
encoder/block_008/layer_000/SelfAttention/k_slot_vc
encoder/block_008/layer_000/SelfAttention/k_slot_vr
encoder/block_008/layer_000/SelfAttention/o_slot_vc
encoder/block_008/layer_000/SelfAttention/o_slot_vr
encoder/block_008/layer_000/SelfAttention/q_slot_vc
encoder/block_008/layer_000/SelfAttention/q_slot_vr
encoder/block_008/layer_000/SelfAttention/v_slot_vc
encoder/block_008/layer_000/SelfAttention/v_slot_vr
encoder/block_008/layer_000/layer_norm/scale_slot_v
encoder/block_008/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_008/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_008/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_008/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_008/layer_001/layer_norm/scale_slot_v
encoder/block_009/layer_000/SelfAttention/k_slot_vc
encoder/block_009/layer_000/SelfAttention/k_slot_vr
encoder/block_009/layer_000/SelfAttention/o_slot_vc
encoder/block_009/layer_000/SelfAttention/o_slot_vr
encoder/block_009/layer_000/SelfAttention/q_slot_vc
encoder/block_009/layer_000/SelfAttention/q_slot_vr
encoder/block_009/layer_000/SelfAttention/v_slot_vc
encoder/block_009/layer_000/SelfAttention/v_slot_vr
encoder/block_009/layer_000/layer_norm/scale_slot_v
encoder/block_009/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_009/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_009/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_009/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_009/layer_001/layer_norm/scale_slot_v
encoder/block_010/layer_000/SelfAttention/k_slot_vc
encoder/block_010/layer_000/SelfAttention/k_slot_vr
encoder/block_010/layer_000/SelfAttention/o_slot_vc
encoder/block_010/layer_000/SelfAttention/o_slot_vr
encoder/block_010/layer_000/SelfAttention/q_slot_vc
encoder/block_010/layer_000/SelfAttention/q_slot_vr
encoder/block_010/layer_000/SelfAttention/v_slot_vc
encoder/block_010/layer_000/SelfAttention/v_slot_vr
encoder/block_010/layer_000/layer_norm/scale_slot_v
encoder/block_010/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_010/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_010/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_010/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_010/layer_001/layer_norm/scale_slot_v
encoder/block_011/layer_000/SelfAttention/k_slot_vc
encoder/block_011/layer_000/SelfAttention/k_slot_vr
encoder/block_011/layer_000/SelfAttention/o_slot_vc
encoder/block_011/layer_000/SelfAttention/o_slot_vr
encoder/block_011/layer_000/SelfAttention/q_slot_vc
encoder/block_011/layer_000/SelfAttention/q_slot_vr
encoder/block_011/layer_000/SelfAttention/v_slot_vc
encoder/block_011/layer_000/SelfAttention/v_slot_vr
encoder/block_011/layer_000/layer_norm/scale_slot_v
encoder/block_011/layer_001/DenseReluDense/wi/kernel_slot_vc
encoder/block_011/layer_001/DenseReluDense/wi/kernel_slot_vr
encoder/block_011/layer_001/DenseReluDense/wo/kernel_slot_vc
encoder/block_011/layer_001/DenseReluDense/wo/kernel_slot_vr
encoder/block_011/layer_001/layer_norm/scale_slot_v
encoder/final_layer_norm/scale_slot_v
INFO:tensorflow:Variables in graph but not in gs://t5-data/pretrained_models/base/model.ckpt-999900:
INFO:tensorflow:
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:TPU job name worker
INFO:tensorflow:Starting the session.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
WARNING:tensorflow:From /home/marcospiau123/.local/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py:767: Variable.load (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
Instructions for updating:
Prefer Variable.assign which has equivalent behavior in 2.X.
INFO:tensorflow:Initialized dataset iterators in 1 seconds
INFO:tensorflow:Installing graceful shutdown hook.
2020-07-26 19:53:38.590184: W tensorflow/core/distributed_runtime/rpc/grpc_session.cc:373] GrpcSession::ListDevices will initialize the session with an empty graph and other defaults because the session has not yet been created.
INFO:tensorflow:Creating heartbeat manager for ['/job:worker/replica:0/task:0/device:CPU:0']
INFO:tensorflow:Configuring worker heartbeat: shutdown_mode: WAIT_FOR_COORDINATOR

INFO:tensorflow:Starting infeed thread controller.
INFO:tensorflow:Starting outfeed thread controller.
INFO:tensorflow:Before copy master to slices.
INFO:tensorflow:Done with copy master to slices.
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 999900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 999900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-999900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 999900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (0, 0)
INFO:tensorflow:loss = 0.3671875, step = 1000000
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (1, 37)
INFO:tensorflow:loss = 0.31445312, step = 1000100 (45.773 sec)
INFO:tensorflow:global_step/sec: 2.18469
INFO:tensorflow:examples/sec: 279.641
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (2, 84)
INFO:tensorflow:loss = 0.26171875, step = 1000200 (41.178 sec)
INFO:tensorflow:global_step/sec: 2.42847
INFO:tensorflow:examples/sec: 310.844
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.25585938, step = 1000300 (40.708 sec)
INFO:tensorflow:global_step/sec: 2.45649
INFO:tensorflow:examples/sec: 314.431
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (4, 32)
INFO:tensorflow:loss = 0.23828125, step = 1000400 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45666
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (5, 79)
INFO:tensorflow:loss = 0.20605469, step = 1000500 (41.154 sec)
INFO:tensorflow:global_step/sec: 2.42988
INFO:tensorflow:examples/sec: 311.025
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1000500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1000500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1000500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1000500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (6, 99)
INFO:tensorflow:loss = 0.20507812, step = 1000600 (52.198 sec)
INFO:tensorflow:global_step/sec: 1.91578
INFO:tensorflow:examples/sec: 245.22
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.20898438, step = 1000700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.455
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (8, 45)
INFO:tensorflow:loss = 0.20703125, step = 1000800 (41.295 sec)
INFO:tensorflow:global_step/sec: 2.42162
INFO:tensorflow:examples/sec: 309.968
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (9, 93)
INFO:tensorflow:loss = 0.19824219, step = 1000900 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.455
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.20214844, step = 1001000 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45659
INFO:tensorflow:examples/sec: 314.443
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (11, 40)
INFO:tensorflow:loss = 0.18652344, step = 1001100 (41.247 sec)
INFO:tensorflow:global_step/sec: 2.42445
INFO:tensorflow:examples/sec: 310.329
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1001100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1001100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1001100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1001100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (12, 59)
INFO:tensorflow:loss = 0.17480469, step = 1001200 (52.479 sec)
INFO:tensorflow:global_step/sec: 1.90554
INFO:tensorflow:examples/sec: 243.909
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.18261719, step = 1001300 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45663
INFO:tensorflow:examples/sec: 314.449
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (14, 6)
INFO:tensorflow:loss = 0.17773438, step = 1001400 (41.170 sec)
INFO:tensorflow:global_step/sec: 2.42895
INFO:tensorflow:examples/sec: 310.906
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (15, 54)
INFO:tensorflow:loss = 0.18554688, step = 1001500 (40.708 sec)
INFO:tensorflow:global_step/sec: 2.45653
INFO:tensorflow:examples/sec: 314.436
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.18066406, step = 1001600 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.459
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (17, 1)
INFO:tensorflow:loss = 0.18066406, step = 1001700 (41.239 sec)
INFO:tensorflow:global_step/sec: 2.4249
INFO:tensorflow:examples/sec: 310.387
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1001700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1001700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1001700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1001700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (18, 17)
INFO:tensorflow:loss = 0.17578125, step = 1001800 (53.702 sec)
INFO:tensorflow:global_step/sec: 1.86214
INFO:tensorflow:examples/sec: 238.354
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (19, 65)
INFO:tensorflow:loss = 0.1796875, step = 1001900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.4568
INFO:tensorflow:examples/sec: 314.471
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.171875, step = 1002000 (41.188 sec)
INFO:tensorflow:global_step/sec: 2.4279
INFO:tensorflow:examples/sec: 310.772
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (21, 12)
INFO:tensorflow:loss = 0.16992188, step = 1002100 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (22, 60)
INFO:tensorflow:loss = 0.16601562, step = 1002200 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45662
INFO:tensorflow:examples/sec: 314.447
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.16796875, step = 1002300 (41.285 sec)
INFO:tensorflow:global_step/sec: 2.4222
INFO:tensorflow:examples/sec: 310.041
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1002300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1002300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1002300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1002300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (24, 0)
INFO:tensorflow:loss = 0.17480469, step = 1002400 (52.478 sec)
INFO:tensorflow:global_step/sec: 1.90558
INFO:tensorflow:examples/sec: 243.915
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (25, 48)
INFO:tensorflow:loss = 0.17285156, step = 1002500 (40.708 sec)
INFO:tensorflow:global_step/sec: 2.45648
INFO:tensorflow:examples/sec: 314.43
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (26, 95)
INFO:tensorflow:loss = 0.17285156, step = 1002600 (41.278 sec)
INFO:tensorflow:global_step/sec: 2.4226
INFO:tensorflow:examples/sec: 310.093
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.16796875, step = 1002700 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45663
INFO:tensorflow:examples/sec: 314.449
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (28, 43)
INFO:tensorflow:loss = 0.16796875, step = 1002800 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.461
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (29, 90)
INFO:tensorflow:loss = 0.17675781, step = 1002900 (41.133 sec)
INFO:tensorflow:global_step/sec: 2.43114
INFO:tensorflow:examples/sec: 311.186
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1002900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1002900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1002900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1002900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.18359375, step = 1003000 (52.384 sec)
INFO:tensorflow:global_step/sec: 1.90902
INFO:tensorflow:examples/sec: 244.354
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (31, 9)
INFO:tensorflow:loss = 0.17578125, step = 1003100 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45681
INFO:tensorflow:examples/sec: 314.472
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (32, 54)
INFO:tensorflow:loss = 0.16113281, step = 1003200 (41.929 sec)
INFO:tensorflow:global_step/sec: 2.38499
INFO:tensorflow:examples/sec: 305.279
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.17382812, step = 1003300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45666
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (34, 2)
INFO:tensorflow:loss = 0.16992188, step = 1003400 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45664
INFO:tensorflow:examples/sec: 314.45
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (35, 49)
INFO:tensorflow:loss = 0.1640625, step = 1003500 (41.186 sec)
INFO:tensorflow:global_step/sec: 2.42804
INFO:tensorflow:examples/sec: 310.79
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1003500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1003500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1003500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1003500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (36, 68)
INFO:tensorflow:loss = 0.16015625, step = 1003600 (52.434 sec)
INFO:tensorflow:global_step/sec: 1.90716
INFO:tensorflow:examples/sec: 244.116
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.17480469, step = 1003700 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (38, 15)
INFO:tensorflow:loss = 0.15820312, step = 1003800 (41.136 sec)
INFO:tensorflow:global_step/sec: 2.43094
INFO:tensorflow:examples/sec: 311.16
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (39, 63)
INFO:tensorflow:loss = 0.15527344, step = 1003900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15625, step = 1004000 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (41, 10)
INFO:tensorflow:loss = 0.1640625, step = 1004100 (41.199 sec)
INFO:tensorflow:global_step/sec: 2.42719
INFO:tensorflow:examples/sec: 310.68
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1004100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1004100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1004100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1004100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (42, 30)
INFO:tensorflow:loss = 0.16210938, step = 1004200 (51.961 sec)
INFO:tensorflow:global_step/sec: 1.92453
INFO:tensorflow:examples/sec: 246.34
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (43, 78)
INFO:tensorflow:loss = 0.1484375, step = 1004300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.455
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15820312, step = 1004400 (41.227 sec)
INFO:tensorflow:global_step/sec: 2.4256
INFO:tensorflow:examples/sec: 310.477
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (45, 25)
INFO:tensorflow:loss = 0.1640625, step = 1004500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (46, 73)
INFO:tensorflow:loss = 0.1484375, step = 1004600 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.453
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.16210938, step = 1004700 (41.126 sec)
INFO:tensorflow:global_step/sec: 2.43158
INFO:tensorflow:examples/sec: 311.242
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1004700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1004700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1004700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1004700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (48, 0)
INFO:tensorflow:loss = 0.1640625, step = 1004800 (50.908 sec)
INFO:tensorflow:global_step/sec: 1.96436
INFO:tensorflow:examples/sec: 251.438
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (49, 48)
INFO:tensorflow:loss = 0.15039062, step = 1004900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45666
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (50, 95)
INFO:tensorflow:loss = 0.15625, step = 1005000 (41.201 sec)
INFO:tensorflow:global_step/sec: 2.42712
INFO:tensorflow:examples/sec: 310.671
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.16015625, step = 1005100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45662
INFO:tensorflow:examples/sec: 314.447
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (52, 43)
INFO:tensorflow:loss = 0.14648438, step = 1005200 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (53, 90)
INFO:tensorflow:loss = 0.15234375, step = 1005300 (41.249 sec)
INFO:tensorflow:global_step/sec: 2.42428
INFO:tensorflow:examples/sec: 310.308
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1005300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1005300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1005300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1005300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1005400 (52.839 sec)
INFO:tensorflow:global_step/sec: 1.89254
INFO:tensorflow:examples/sec: 242.246
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (55, 8)
INFO:tensorflow:loss = 0.140625, step = 1005500 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.457
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (56, 55)
INFO:tensorflow:loss = 0.15820312, step = 1005600 (41.247 sec)
INFO:tensorflow:global_step/sec: 2.42441
INFO:tensorflow:examples/sec: 310.324
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15332031, step = 1005700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (58, 3)
INFO:tensorflow:loss = 0.140625, step = 1005800 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45687
INFO:tensorflow:examples/sec: 314.48
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (59, 50)
INFO:tensorflow:loss = 0.15527344, step = 1005900 (41.208 sec)
INFO:tensorflow:global_step/sec: 2.4267
INFO:tensorflow:examples/sec: 310.617
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1005900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1005900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1005900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1005900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (60, 70)
INFO:tensorflow:loss = 0.15039062, step = 1006000 (51.891 sec)
INFO:tensorflow:global_step/sec: 1.92714
INFO:tensorflow:examples/sec: 246.674
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1640625, step = 1006100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (62, 16)
INFO:tensorflow:loss = 0.15039062, step = 1006200 (41.329 sec)
INFO:tensorflow:global_step/sec: 2.4196
INFO:tensorflow:examples/sec: 309.709
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (63, 64)
INFO:tensorflow:loss = 0.1484375, step = 1006300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45662
INFO:tensorflow:examples/sec: 314.448
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15820312, step = 1006400 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45681
INFO:tensorflow:examples/sec: 314.472
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (65, 10)
INFO:tensorflow:loss = 0.15429688, step = 1006500 (41.383 sec)
INFO:tensorflow:global_step/sec: 2.41644
INFO:tensorflow:examples/sec: 309.305
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1006500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1006500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1006500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1006500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (66, 27)
INFO:tensorflow:loss = 0.1484375, step = 1006600 (53.401 sec)
INFO:tensorflow:global_step/sec: 1.87262
INFO:tensorflow:examples/sec: 239.695
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (67, 75)
INFO:tensorflow:loss = 0.15820312, step = 1006700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.455
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15820312, step = 1006800 (41.276 sec)
INFO:tensorflow:global_step/sec: 2.42273
INFO:tensorflow:examples/sec: 310.109
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (69, 21)
INFO:tensorflow:loss = 0.14746094, step = 1006900 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45666
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (70, 69)
INFO:tensorflow:loss = 0.16210938, step = 1007000 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.4568
INFO:tensorflow:examples/sec: 314.471
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1007100 (41.180 sec)
INFO:tensorflow:global_step/sec: 2.42834
INFO:tensorflow:examples/sec: 310.828
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1007100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1007100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1007100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1007100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (72, 0)
INFO:tensorflow:loss = 0.15625, step = 1007200 (52.333 sec)
INFO:tensorflow:global_step/sec: 1.91082
INFO:tensorflow:examples/sec: 244.585
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (73, 48)
INFO:tensorflow:loss = 0.15039062, step = 1007300 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45684
INFO:tensorflow:examples/sec: 314.476
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (74, 95)
INFO:tensorflow:loss = 0.15820312, step = 1007400 (41.225 sec)
INFO:tensorflow:global_step/sec: 2.42575
INFO:tensorflow:examples/sec: 310.496
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1007500 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (76, 43)
INFO:tensorflow:loss = 0.15722656, step = 1007600 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (77, 90)
INFO:tensorflow:loss = 0.15429688, step = 1007700 (41.193 sec)
INFO:tensorflow:global_step/sec: 2.42764
INFO:tensorflow:examples/sec: 310.738
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1007700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1007700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1007700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1007700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15917969, step = 1007800 (52.123 sec)
INFO:tensorflow:global_step/sec: 1.91853
INFO:tensorflow:examples/sec: 245.572
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (79, 10)
INFO:tensorflow:loss = 0.15039062, step = 1007900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45666
INFO:tensorflow:examples/sec: 314.453
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (80, 57)
INFO:tensorflow:loss = 0.14941406, step = 1008000 (41.167 sec)
INFO:tensorflow:global_step/sec: 2.42911
INFO:tensorflow:examples/sec: 310.927
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1008100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.463
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (82, 5)
INFO:tensorflow:loss = 0.15039062, step = 1008200 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.455
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (83, 51)
INFO:tensorflow:loss = 0.14550781, step = 1008300 (41.423 sec)
INFO:tensorflow:global_step/sec: 2.41409
INFO:tensorflow:examples/sec: 309.004
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1008300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1008300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1008300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1008300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (84, 71)
INFO:tensorflow:loss = 0.1484375, step = 1008400 (52.264 sec)
INFO:tensorflow:global_step/sec: 1.91338
INFO:tensorflow:examples/sec: 244.912
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1008500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.468
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (86, 17)
INFO:tensorflow:loss = 0.14746094, step = 1008600 (41.413 sec)
INFO:tensorflow:global_step/sec: 2.41468
INFO:tensorflow:examples/sec: 309.079
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (87, 65)
INFO:tensorflow:loss = 0.14257812, step = 1008700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15429688, step = 1008800 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (89, 11)
INFO:tensorflow:loss = 0.1484375, step = 1008900 (41.320 sec)
INFO:tensorflow:global_step/sec: 2.42013
INFO:tensorflow:examples/sec: 309.777
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1008900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1008900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1008900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1008900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (90, 31)
INFO:tensorflow:loss = 0.15722656, step = 1009000 (52.011 sec)
INFO:tensorflow:global_step/sec: 1.92267
INFO:tensorflow:examples/sec: 246.102
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (91, 79)
INFO:tensorflow:loss = 0.15625, step = 1009100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.16699219, step = 1009200 (41.173 sec)
INFO:tensorflow:global_step/sec: 2.42876
INFO:tensorflow:examples/sec: 310.881
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (93, 26)
INFO:tensorflow:loss = 0.1640625, step = 1009300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (94, 74)
INFO:tensorflow:loss = 0.15625, step = 1009400 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15722656, step = 1009500 (41.260 sec)
INFO:tensorflow:global_step/sec: 2.42364
INFO:tensorflow:examples/sec: 310.225
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1009500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1009500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1009500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1009500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (96, 0)
INFO:tensorflow:loss = 0.1484375, step = 1009600 (52.214 sec)
INFO:tensorflow:global_step/sec: 1.91522
INFO:tensorflow:examples/sec: 245.148
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (97, 48)
INFO:tensorflow:loss = 0.1484375, step = 1009700 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45681
INFO:tensorflow:examples/sec: 314.472
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (98, 95)
INFO:tensorflow:loss = 0.13671875, step = 1009800 (41.283 sec)
INFO:tensorflow:global_step/sec: 2.4223
INFO:tensorflow:examples/sec: 310.054
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1009900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (100, 43)
INFO:tensorflow:loss = 0.15917969, step = 1010000 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (101, 90)
INFO:tensorflow:loss = 0.1484375, step = 1010100 (41.160 sec)
INFO:tensorflow:global_step/sec: 2.42955
INFO:tensorflow:examples/sec: 310.982
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1010100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1010100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1010100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1010100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.16015625, step = 1010200 (51.279 sec)
INFO:tensorflow:global_step/sec: 1.95013
INFO:tensorflow:examples/sec: 249.617
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (103, 12)
INFO:tensorflow:loss = 0.14355469, step = 1010300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.4568
INFO:tensorflow:examples/sec: 314.47
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (104, 58)
INFO:tensorflow:loss = 0.13867188, step = 1010400 (41.305 sec)
INFO:tensorflow:global_step/sec: 2.42097
INFO:tensorflow:examples/sec: 309.884
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15039062, step = 1010500 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.457
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (106, 6)
INFO:tensorflow:loss = 0.14257812, step = 1010600 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (107, 53)
INFO:tensorflow:loss = 0.14941406, step = 1010700 (41.184 sec)
INFO:tensorflow:global_step/sec: 2.4281
INFO:tensorflow:examples/sec: 310.797
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1010700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1010700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1010700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1010700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (108, 71)
INFO:tensorflow:loss = 0.15332031, step = 1010800 (52.749 sec)
INFO:tensorflow:global_step/sec: 1.89579
INFO:tensorflow:examples/sec: 242.661
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14550781, step = 1010900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (110, 18)
INFO:tensorflow:loss = 0.15625, step = 1011000 (41.157 sec)
INFO:tensorflow:global_step/sec: 2.42973
INFO:tensorflow:examples/sec: 311.006
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (111, 66)
INFO:tensorflow:loss = 0.14648438, step = 1011100 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45685
INFO:tensorflow:examples/sec: 314.476
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15136719, step = 1011200 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (113, 13)
INFO:tensorflow:loss = 0.14257812, step = 1011300 (41.233 sec)
INFO:tensorflow:global_step/sec: 2.42526
INFO:tensorflow:examples/sec: 310.433
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1011300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1011300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1011300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1011300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (114, 29)
INFO:tensorflow:loss = 0.14746094, step = 1011400 (53.827 sec)
INFO:tensorflow:global_step/sec: 1.8578
INFO:tensorflow:examples/sec: 237.799
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (115, 77)
INFO:tensorflow:loss = 0.140625, step = 1011500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13964844, step = 1011600 (41.155 sec)
INFO:tensorflow:global_step/sec: 2.42982
INFO:tensorflow:examples/sec: 311.017
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (117, 24)
INFO:tensorflow:loss = 0.13671875, step = 1011700 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (118, 72)
INFO:tensorflow:loss = 0.15429688, step = 1011800 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1011900 (41.221 sec)
INFO:tensorflow:global_step/sec: 2.42594
INFO:tensorflow:examples/sec: 310.52
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1011900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1011900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1011900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1011900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (120, 0)
INFO:tensorflow:loss = 0.140625, step = 1012000 (52.485 sec)
INFO:tensorflow:global_step/sec: 1.90531
INFO:tensorflow:examples/sec: 243.88
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (121, 48)
INFO:tensorflow:loss = 0.15429688, step = 1012100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (122, 94)
INFO:tensorflow:loss = 0.1484375, step = 1012200 (41.285 sec)
INFO:tensorflow:global_step/sec: 2.42216
INFO:tensorflow:examples/sec: 310.037
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14355469, step = 1012300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45666
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (124, 42)
INFO:tensorflow:loss = 0.15136719, step = 1012400 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (125, 89)
INFO:tensorflow:loss = 0.14160156, step = 1012500 (41.175 sec)
INFO:tensorflow:global_step/sec: 2.42868
INFO:tensorflow:examples/sec: 310.871
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1012500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1012500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1012500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1012500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1012600 (53.524 sec)
INFO:tensorflow:global_step/sec: 1.86832
INFO:tensorflow:examples/sec: 239.145
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (127, 5)
INFO:tensorflow:loss = 0.13867188, step = 1012700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.461
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (128, 52)
INFO:tensorflow:loss = 0.1484375, step = 1012800 (41.167 sec)
INFO:tensorflow:global_step/sec: 2.4291
INFO:tensorflow:examples/sec: 310.925
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.140625, step = 1012900 (40.708 sec)
INFO:tensorflow:global_step/sec: 2.45654
INFO:tensorflow:examples/sec: 314.438
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (130, 0)
INFO:tensorflow:loss = 0.15234375, step = 1013000 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (131, 47)
INFO:tensorflow:loss = 0.140625, step = 1013100 (41.207 sec)
INFO:tensorflow:global_step/sec: 2.42676
INFO:tensorflow:examples/sec: 310.626
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1013100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1013100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1013100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1013100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (132, 69)
INFO:tensorflow:loss = 0.14257812, step = 1013200 (51.446 sec)
INFO:tensorflow:global_step/sec: 1.94378
INFO:tensorflow:examples/sec: 248.804
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14160156, step = 1013300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45666
INFO:tensorflow:examples/sec: 314.453
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (134, 16)
INFO:tensorflow:loss = 0.16601562, step = 1013400 (41.184 sec)
INFO:tensorflow:global_step/sec: 2.42809
INFO:tensorflow:examples/sec: 310.795
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (135, 64)
INFO:tensorflow:loss = 0.14746094, step = 1013500 (40.708 sec)
INFO:tensorflow:global_step/sec: 2.45652
INFO:tensorflow:examples/sec: 314.434
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14160156, step = 1013600 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (137, 11)
INFO:tensorflow:loss = 0.14257812, step = 1013700 (41.222 sec)
INFO:tensorflow:global_step/sec: 2.42587
INFO:tensorflow:examples/sec: 310.511
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1013700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1013700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1013700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1013700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (138, 33)
INFO:tensorflow:loss = 0.13476562, step = 1013800 (51.141 sec)
INFO:tensorflow:global_step/sec: 1.9554
INFO:tensorflow:examples/sec: 250.291
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (139, 81)
INFO:tensorflow:loss = 0.1484375, step = 1013900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13671875, step = 1014000 (41.270 sec)
INFO:tensorflow:global_step/sec: 2.42303
INFO:tensorflow:examples/sec: 310.148
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (141, 27)
INFO:tensorflow:loss = 0.13867188, step = 1014100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (142, 75)
INFO:tensorflow:loss = 0.13867188, step = 1014200 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.463
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1484375, step = 1014300 (41.217 sec)
INFO:tensorflow:global_step/sec: 2.42621
INFO:tensorflow:examples/sec: 310.555
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1014300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1014300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1014300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1014300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (144, 0)
INFO:tensorflow:loss = 0.13476562, step = 1014400 (52.024 sec)
INFO:tensorflow:global_step/sec: 1.92218
INFO:tensorflow:examples/sec: 246.039
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (145, 48)
INFO:tensorflow:loss = 0.140625, step = 1014500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.468
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (146, 94)
INFO:tensorflow:loss = 0.140625, step = 1014600 (41.333 sec)
INFO:tensorflow:global_step/sec: 2.41938
INFO:tensorflow:examples/sec: 309.68
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1014700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.456
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (148, 42)
INFO:tensorflow:loss = 0.140625, step = 1014800 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (149, 89)
INFO:tensorflow:loss = 0.14160156, step = 1014900 (41.252 sec)
INFO:tensorflow:global_step/sec: 2.42409
INFO:tensorflow:examples/sec: 310.284
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1014900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1014900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1014900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1014900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1484375, step = 1015000 (52.139 sec)
INFO:tensorflow:global_step/sec: 1.91794
INFO:tensorflow:examples/sec: 245.497
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (151, 9)
INFO:tensorflow:loss = 0.14648438, step = 1015100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.455
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (152, 56)
INFO:tensorflow:loss = 0.14648438, step = 1015200 (41.175 sec)
INFO:tensorflow:global_step/sec: 2.4287
INFO:tensorflow:examples/sec: 310.874
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1484375, step = 1015300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.451
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (154, 4)
INFO:tensorflow:loss = 0.14453125, step = 1015400 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (155, 50)
INFO:tensorflow:loss = 0.14355469, step = 1015500 (41.421 sec)
INFO:tensorflow:global_step/sec: 2.41422
INFO:tensorflow:examples/sec: 309.021
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1015500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1015500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1015500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1015500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (156, 71)
INFO:tensorflow:loss = 0.14453125, step = 1015600 (51.483 sec)
INFO:tensorflow:global_step/sec: 1.94243
INFO:tensorflow:examples/sec: 248.631
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1015700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (158, 18)
INFO:tensorflow:loss = 0.14257812, step = 1015800 (41.154 sec)
INFO:tensorflow:global_step/sec: 2.4299
INFO:tensorflow:examples/sec: 311.027
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (159, 66)
INFO:tensorflow:loss = 0.13574219, step = 1015900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13671875, step = 1016000 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45663
INFO:tensorflow:examples/sec: 314.449
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (161, 13)
INFO:tensorflow:loss = 0.1484375, step = 1016100 (41.228 sec)
INFO:tensorflow:global_step/sec: 2.42555
INFO:tensorflow:examples/sec: 310.471
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1016100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1016100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1016100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1016100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (162, 31)
INFO:tensorflow:loss = 0.13964844, step = 1016200 (52.802 sec)
INFO:tensorflow:global_step/sec: 1.89388
INFO:tensorflow:examples/sec: 242.417
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (163, 79)
INFO:tensorflow:loss = 0.140625, step = 1016300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45661
INFO:tensorflow:examples/sec: 314.447
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14746094, step = 1016400 (41.117 sec)
INFO:tensorflow:global_step/sec: 2.43209
INFO:tensorflow:examples/sec: 311.308
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (165, 26)
INFO:tensorflow:loss = 0.140625, step = 1016500 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45685
INFO:tensorflow:examples/sec: 314.477
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (166, 74)
INFO:tensorflow:loss = 0.12792969, step = 1016600 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.457
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1016700 (41.226 sec)
INFO:tensorflow:global_step/sec: 2.42567
INFO:tensorflow:examples/sec: 310.486
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1016700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1016700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1016700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1016700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (168, 0)
INFO:tensorflow:loss = 0.14257812, step = 1016800 (51.403 sec)
INFO:tensorflow:global_step/sec: 1.94542
INFO:tensorflow:examples/sec: 249.014
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (169, 48)
INFO:tensorflow:loss = 0.13085938, step = 1016900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (170, 95)
INFO:tensorflow:loss = 0.13964844, step = 1017000 (41.178 sec)
INFO:tensorflow:global_step/sec: 2.42847
INFO:tensorflow:examples/sec: 310.845
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1017100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45666
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (172, 43)
INFO:tensorflow:loss = 0.13867188, step = 1017200 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.457
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (173, 90)
INFO:tensorflow:loss = 0.140625, step = 1017300 (41.261 sec)
INFO:tensorflow:global_step/sec: 2.42361
INFO:tensorflow:examples/sec: 310.222
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1017300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1017300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1017300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1017300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15429688, step = 1017400 (51.939 sec)
INFO:tensorflow:global_step/sec: 1.92534
INFO:tensorflow:examples/sec: 246.443
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (175, 10)
INFO:tensorflow:loss = 0.1484375, step = 1017500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (176, 57)
INFO:tensorflow:loss = 0.15136719, step = 1017600 (41.204 sec)
INFO:tensorflow:global_step/sec: 2.42691
INFO:tensorflow:examples/sec: 310.645
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15039062, step = 1017700 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (178, 5)
INFO:tensorflow:loss = 0.14453125, step = 1017800 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45687
INFO:tensorflow:examples/sec: 314.48
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (179, 52)
INFO:tensorflow:loss = 0.14257812, step = 1017900 (41.278 sec)
INFO:tensorflow:global_step/sec: 2.42259
INFO:tensorflow:examples/sec: 310.091
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1017900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1017900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1017900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1017900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (180, 73)
INFO:tensorflow:loss = 0.15625, step = 1018000 (51.526 sec)
INFO:tensorflow:global_step/sec: 1.94074
INFO:tensorflow:examples/sec: 248.415
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15039062, step = 1018100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.457
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (182, 20)
INFO:tensorflow:loss = 0.15136719, step = 1018200 (41.147 sec)
INFO:tensorflow:global_step/sec: 2.43033
INFO:tensorflow:examples/sec: 311.082
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (183, 68)
INFO:tensorflow:loss = 0.14648438, step = 1018300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1484375, step = 1018400 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (185, 15)
INFO:tensorflow:loss = 0.1484375, step = 1018500 (41.134 sec)
INFO:tensorflow:global_step/sec: 2.4311
INFO:tensorflow:examples/sec: 311.181
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1018500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1018500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1018500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1018500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (186, 34)
INFO:tensorflow:loss = 0.14257812, step = 1018600 (52.403 sec)
INFO:tensorflow:global_step/sec: 1.90828
INFO:tensorflow:examples/sec: 244.26
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (187, 82)
INFO:tensorflow:loss = 0.14453125, step = 1018700 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1018800 (41.219 sec)
INFO:tensorflow:global_step/sec: 2.42607
INFO:tensorflow:examples/sec: 310.537
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (189, 29)
INFO:tensorflow:loss = 0.13867188, step = 1018900 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.451
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (190, 77)
INFO:tensorflow:loss = 0.13867188, step = 1019000 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45681
INFO:tensorflow:examples/sec: 314.472
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1019100 (41.216 sec)
INFO:tensorflow:global_step/sec: 2.42619
INFO:tensorflow:examples/sec: 310.552
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1019100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1019100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1019100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1019100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (192, 0)
INFO:tensorflow:loss = 0.14941406, step = 1019200 (52.781 sec)
INFO:tensorflow:global_step/sec: 1.89463
INFO:tensorflow:examples/sec: 242.513
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (193, 48)
INFO:tensorflow:loss = 0.14453125, step = 1019300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (194, 95)
INFO:tensorflow:loss = 0.1484375, step = 1019400 (41.175 sec)
INFO:tensorflow:global_step/sec: 2.42865
INFO:tensorflow:examples/sec: 310.867
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14746094, step = 1019500 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (196, 43)
INFO:tensorflow:loss = 0.15429688, step = 1019600 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.463
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (197, 90)
INFO:tensorflow:loss = 0.140625, step = 1019700 (41.227 sec)
INFO:tensorflow:global_step/sec: 2.42561
INFO:tensorflow:examples/sec: 310.478
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1019700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1019700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1019700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1019700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1484375, step = 1019800 (51.715 sec)
INFO:tensorflow:global_step/sec: 1.93369
INFO:tensorflow:examples/sec: 247.513
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (199, 11)
INFO:tensorflow:loss = 0.14550781, step = 1019900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.461
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (200, 58)
INFO:tensorflow:loss = 0.15429688, step = 1020000 (41.174 sec)
INFO:tensorflow:global_step/sec: 2.42874
INFO:tensorflow:examples/sec: 310.878
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13671875, step = 1020100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.459
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (202, 6)
INFO:tensorflow:loss = 0.13964844, step = 1020200 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (203, 53)
INFO:tensorflow:loss = 0.140625, step = 1020300 (41.213 sec)
INFO:tensorflow:global_step/sec: 2.42643
INFO:tensorflow:examples/sec: 310.583
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1020300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1020300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1020300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1020300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (204, 73)
INFO:tensorflow:loss = 0.14257812, step = 1020400 (52.004 sec)
INFO:tensorflow:global_step/sec: 1.92293
INFO:tensorflow:examples/sec: 246.136
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13964844, step = 1020500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.459
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (206, 20)
INFO:tensorflow:loss = 0.14355469, step = 1020600 (41.197 sec)
INFO:tensorflow:global_step/sec: 2.42735
INFO:tensorflow:examples/sec: 310.701
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (207, 68)
INFO:tensorflow:loss = 0.14160156, step = 1020700 (40.707 sec)
INFO:tensorflow:global_step/sec: 2.45657
INFO:tensorflow:examples/sec: 314.441
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1020800 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.4568
INFO:tensorflow:examples/sec: 314.471
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (209, 15)
INFO:tensorflow:loss = 0.15234375, step = 1020900 (41.171 sec)
INFO:tensorflow:global_step/sec: 2.42889
INFO:tensorflow:examples/sec: 310.898
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1020900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1020900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1020900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1020900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (210, 34)
INFO:tensorflow:loss = 0.13867188, step = 1021000 (52.487 sec)
INFO:tensorflow:global_step/sec: 1.90521
INFO:tensorflow:examples/sec: 243.867
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (211, 82)
INFO:tensorflow:loss = 0.13769531, step = 1021100 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1021200 (41.222 sec)
INFO:tensorflow:global_step/sec: 2.42589
INFO:tensorflow:examples/sec: 310.514
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (213, 29)
INFO:tensorflow:loss = 0.14160156, step = 1021300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.461
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (214, 77)
INFO:tensorflow:loss = 0.14257812, step = 1021400 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14355469, step = 1021500 (41.238 sec)
INFO:tensorflow:global_step/sec: 2.42494
INFO:tensorflow:examples/sec: 310.392
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1021500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1021500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1021500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1021500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (216, 0)
INFO:tensorflow:loss = 0.14257812, step = 1021600 (52.681 sec)
INFO:tensorflow:global_step/sec: 1.89822
INFO:tensorflow:examples/sec: 242.972
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (217, 48)
INFO:tensorflow:loss = 0.14453125, step = 1021700 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (218, 95)
INFO:tensorflow:loss = 0.140625, step = 1021800 (41.252 sec)
INFO:tensorflow:global_step/sec: 2.42413
INFO:tensorflow:examples/sec: 310.288
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1021900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.456
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (220, 43)
INFO:tensorflow:loss = 0.14453125, step = 1022000 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (221, 90)
INFO:tensorflow:loss = 0.14453125, step = 1022100 (41.225 sec)
INFO:tensorflow:global_step/sec: 2.42574
INFO:tensorflow:examples/sec: 310.494
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1022100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1022100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1022100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1022100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15234375, step = 1022200 (52.333 sec)
INFO:tensorflow:global_step/sec: 1.91083
INFO:tensorflow:examples/sec: 244.586
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (223, 9)
INFO:tensorflow:loss = 0.13867188, step = 1022300 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45685
INFO:tensorflow:examples/sec: 314.476
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (224, 55)
INFO:tensorflow:loss = 0.14550781, step = 1022400 (41.430 sec)
INFO:tensorflow:global_step/sec: 2.41371
INFO:tensorflow:examples/sec: 308.955
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1022500 (40.707 sec)
INFO:tensorflow:global_step/sec: 2.45658
INFO:tensorflow:examples/sec: 314.443
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (226, 3)
INFO:tensorflow:loss = 0.14257812, step = 1022600 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (227, 50)
INFO:tensorflow:loss = 0.14257812, step = 1022700 (41.168 sec)
INFO:tensorflow:global_step/sec: 2.42903
INFO:tensorflow:examples/sec: 310.916
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1022700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1022700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1022700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1022700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (228, 70)
INFO:tensorflow:loss = 0.15234375, step = 1022800 (52.259 sec)
INFO:tensorflow:global_step/sec: 1.91356
INFO:tensorflow:examples/sec: 244.936
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13964844, step = 1022900 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (230, 17)
INFO:tensorflow:loss = 0.14257812, step = 1023000 (41.178 sec)
INFO:tensorflow:global_step/sec: 2.42845
INFO:tensorflow:examples/sec: 310.842
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (231, 65)
INFO:tensorflow:loss = 0.14941406, step = 1023100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14160156, step = 1023200 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45686
INFO:tensorflow:examples/sec: 314.478
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (233, 12)
INFO:tensorflow:loss = 0.1484375, step = 1023300 (41.196 sec)
INFO:tensorflow:global_step/sec: 2.42741
INFO:tensorflow:examples/sec: 310.709
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1023300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1023300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1023300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1023300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (234, 32)
INFO:tensorflow:loss = 0.14257812, step = 1023400 (52.223 sec)
INFO:tensorflow:global_step/sec: 1.91487
INFO:tensorflow:examples/sec: 245.103
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (235, 80)
INFO:tensorflow:loss = 0.14648438, step = 1023500 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45685
INFO:tensorflow:examples/sec: 314.477
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1023600 (41.199 sec)
INFO:tensorflow:global_step/sec: 2.42725
INFO:tensorflow:examples/sec: 310.688
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (237, 27)
INFO:tensorflow:loss = 0.15039062, step = 1023700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (238, 75)
INFO:tensorflow:loss = 0.14355469, step = 1023800 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1023900 (41.115 sec)
INFO:tensorflow:global_step/sec: 2.43222
INFO:tensorflow:examples/sec: 311.324
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1023900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1023900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1023900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1023900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (240, 0)
INFO:tensorflow:loss = 0.14746094, step = 1024000 (51.349 sec)
INFO:tensorflow:global_step/sec: 1.94743
INFO:tensorflow:examples/sec: 249.271
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (241, 48)
INFO:tensorflow:loss = 0.14648438, step = 1024100 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45681
INFO:tensorflow:examples/sec: 314.472
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (242, 95)
INFO:tensorflow:loss = 0.13867188, step = 1024200 (41.178 sec)
INFO:tensorflow:global_step/sec: 2.42848
INFO:tensorflow:examples/sec: 310.845
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1024300 (40.707 sec)
INFO:tensorflow:global_step/sec: 2.45664
INFO:tensorflow:examples/sec: 314.45
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (244, 43)
INFO:tensorflow:loss = 0.14648438, step = 1024400 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (245, 90)
INFO:tensorflow:loss = 0.14355469, step = 1024500 (41.155 sec)
INFO:tensorflow:global_step/sec: 2.42984
INFO:tensorflow:examples/sec: 311.02
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1024500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1024500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1024500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1024500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.140625, step = 1024600 (52.541 sec)
INFO:tensorflow:global_step/sec: 1.90329
INFO:tensorflow:examples/sec: 243.621
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (247, 9)
INFO:tensorflow:loss = 0.13671875, step = 1024700 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45686
INFO:tensorflow:examples/sec: 314.478
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (248, 55)
INFO:tensorflow:loss = 0.13671875, step = 1024800 (41.381 sec)
INFO:tensorflow:global_step/sec: 2.41659
INFO:tensorflow:examples/sec: 309.324
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15234375, step = 1024900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (250, 3)
INFO:tensorflow:loss = 0.1484375, step = 1025000 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.463
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (251, 50)
INFO:tensorflow:loss = 0.14550781, step = 1025100 (41.271 sec)
INFO:tensorflow:global_step/sec: 2.42302
INFO:tensorflow:examples/sec: 310.146
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1025100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1025100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1025100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1025100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (252, 67)
INFO:tensorflow:loss = 0.14746094, step = 1025200 (53.112 sec)
INFO:tensorflow:global_step/sec: 1.8828
INFO:tensorflow:examples/sec: 240.998
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1025300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.468
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (254, 14)
INFO:tensorflow:loss = 0.15234375, step = 1025400 (41.146 sec)
INFO:tensorflow:global_step/sec: 2.43038
INFO:tensorflow:examples/sec: 311.089
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (255, 62)
INFO:tensorflow:loss = 0.14746094, step = 1025500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1484375, step = 1025600 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (257, 9)
INFO:tensorflow:loss = 0.140625, step = 1025700 (41.167 sec)
INFO:tensorflow:global_step/sec: 2.42912
INFO:tensorflow:examples/sec: 310.928
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1025700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1025700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1025700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1025700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (258, 30)
INFO:tensorflow:loss = 0.140625, step = 1025800 (51.836 sec)
INFO:tensorflow:global_step/sec: 1.92917
INFO:tensorflow:examples/sec: 246.933
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (259, 78)
INFO:tensorflow:loss = 0.140625, step = 1025900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1026000 (41.280 sec)
INFO:tensorflow:global_step/sec: 2.4225
INFO:tensorflow:examples/sec: 310.08
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (261, 24)
INFO:tensorflow:loss = 0.15527344, step = 1026100 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45681
INFO:tensorflow:examples/sec: 314.471
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (262, 72)
INFO:tensorflow:loss = 0.14453125, step = 1026200 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45682
INFO:tensorflow:examples/sec: 314.473
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1026300 (41.270 sec)
INFO:tensorflow:global_step/sec: 2.42306
INFO:tensorflow:examples/sec: 310.151
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1026300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1026300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1026300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1026300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (264, 0)
INFO:tensorflow:loss = 0.14355469, step = 1026400 (51.517 sec)
INFO:tensorflow:global_step/sec: 1.94111
INFO:tensorflow:examples/sec: 248.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (265, 48)
INFO:tensorflow:loss = 0.13574219, step = 1026500 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.456
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (266, 95)
INFO:tensorflow:loss = 0.140625, step = 1026600 (41.176 sec)
INFO:tensorflow:global_step/sec: 2.42855
INFO:tensorflow:examples/sec: 310.855
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1026700 (40.707 sec)
INFO:tensorflow:global_step/sec: 2.45663
INFO:tensorflow:examples/sec: 314.449
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (268, 43)
INFO:tensorflow:loss = 0.140625, step = 1026800 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.463
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (269, 90)
INFO:tensorflow:loss = 0.13867188, step = 1026900 (41.200 sec)
INFO:tensorflow:global_step/sec: 2.42719
INFO:tensorflow:examples/sec: 310.68
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1026900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1026900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1026900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1026900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14355469, step = 1027000 (53.039 sec)
INFO:tensorflow:global_step/sec: 1.88543
INFO:tensorflow:examples/sec: 241.334
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (271, 8)
INFO:tensorflow:loss = 0.15136719, step = 1027100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (272, 55)
INFO:tensorflow:loss = 0.14160156, step = 1027200 (41.206 sec)
INFO:tensorflow:global_step/sec: 2.42688
INFO:tensorflow:examples/sec: 310.64
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15039062, step = 1027300 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.461
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (274, 3)
INFO:tensorflow:loss = 0.14648438, step = 1027400 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.461
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (275, 50)
INFO:tensorflow:loss = 0.14257812, step = 1027500 (41.258 sec)
INFO:tensorflow:global_step/sec: 2.42376
INFO:tensorflow:examples/sec: 310.241
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1027500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1027500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1027500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1027500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (276, 69)
INFO:tensorflow:loss = 0.1484375, step = 1027600 (52.610 sec)
INFO:tensorflow:global_step/sec: 1.90078
INFO:tensorflow:examples/sec: 243.3
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1027700 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (278, 16)
INFO:tensorflow:loss = 0.14257812, step = 1027800 (41.131 sec)
INFO:tensorflow:global_step/sec: 2.43125
INFO:tensorflow:examples/sec: 311.201
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (279, 64)
INFO:tensorflow:loss = 0.14648438, step = 1027900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15039062, step = 1028000 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (281, 11)
INFO:tensorflow:loss = 0.1484375, step = 1028100 (41.228 sec)
INFO:tensorflow:global_step/sec: 2.42555
INFO:tensorflow:examples/sec: 310.47
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1028100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1028100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1028100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1028100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (282, 31)
INFO:tensorflow:loss = 0.14648438, step = 1028200 (52.021 sec)
INFO:tensorflow:global_step/sec: 1.92232
INFO:tensorflow:examples/sec: 246.057
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (283, 79)
INFO:tensorflow:loss = 0.13964844, step = 1028300 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.451
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.140625, step = 1028400 (41.143 sec)
INFO:tensorflow:global_step/sec: 2.43058
INFO:tensorflow:examples/sec: 311.114
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (285, 26)
INFO:tensorflow:loss = 0.14550781, step = 1028500 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (286, 74)
INFO:tensorflow:loss = 0.1484375, step = 1028600 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45686
INFO:tensorflow:examples/sec: 314.478
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.140625, step = 1028700 (41.188 sec)
INFO:tensorflow:global_step/sec: 2.42789
INFO:tensorflow:examples/sec: 310.77
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1028700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1028700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1028700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1028700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (288, 0)
INFO:tensorflow:loss = 0.15039062, step = 1028800 (52.070 sec)
INFO:tensorflow:global_step/sec: 1.92049
INFO:tensorflow:examples/sec: 245.823
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (289, 48)
INFO:tensorflow:loss = 0.13085938, step = 1028900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (290, 95)
INFO:tensorflow:loss = 0.15234375, step = 1029000 (41.245 sec)
INFO:tensorflow:global_step/sec: 2.4245
INFO:tensorflow:examples/sec: 310.336
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1029100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.457
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (292, 43)
INFO:tensorflow:loss = 0.140625, step = 1029200 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.4568
INFO:tensorflow:examples/sec: 314.47
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (293, 90)
INFO:tensorflow:loss = 0.14746094, step = 1029300 (41.268 sec)
INFO:tensorflow:global_step/sec: 2.4232
INFO:tensorflow:examples/sec: 310.169
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1029300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1029300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1029300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1029300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1029400 (52.133 sec)
INFO:tensorflow:global_step/sec: 1.91818
INFO:tensorflow:examples/sec: 245.528
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (295, 10)
INFO:tensorflow:loss = 0.13964844, step = 1029500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (296, 57)
INFO:tensorflow:loss = 0.14550781, step = 1029600 (41.211 sec)
INFO:tensorflow:global_step/sec: 2.42653
INFO:tensorflow:examples/sec: 310.596
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1029700 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (298, 5)
INFO:tensorflow:loss = 0.13085938, step = 1029800 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45684
INFO:tensorflow:examples/sec: 314.476
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (299, 52)
INFO:tensorflow:loss = 0.14453125, step = 1029900 (41.127 sec)
INFO:tensorflow:global_step/sec: 2.43152
INFO:tensorflow:examples/sec: 311.235
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1029900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1029900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1029900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1029900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (300, 71)
INFO:tensorflow:loss = 0.13476562, step = 1030000 (52.456 sec)
INFO:tensorflow:global_step/sec: 1.90635
INFO:tensorflow:examples/sec: 244.013
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13964844, step = 1030100 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45682
INFO:tensorflow:examples/sec: 314.473
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (302, 18)
INFO:tensorflow:loss = 0.140625, step = 1030200 (41.181 sec)
INFO:tensorflow:global_step/sec: 2.42831
INFO:tensorflow:examples/sec: 310.824
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (303, 66)
INFO:tensorflow:loss = 0.13378906, step = 1030300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.140625, step = 1030400 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (305, 13)
INFO:tensorflow:loss = 0.13085938, step = 1030500 (41.209 sec)
INFO:tensorflow:global_step/sec: 2.42665
INFO:tensorflow:examples/sec: 310.611
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1030500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1030500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1030500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1030500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (306, 32)
INFO:tensorflow:loss = 0.14453125, step = 1030600 (52.398 sec)
INFO:tensorflow:global_step/sec: 1.9085
INFO:tensorflow:examples/sec: 244.288
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (307, 80)
INFO:tensorflow:loss = 0.13085938, step = 1030700 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13964844, step = 1030800 (41.217 sec)
INFO:tensorflow:global_step/sec: 2.4262
INFO:tensorflow:examples/sec: 310.554
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (309, 27)
INFO:tensorflow:loss = 0.13476562, step = 1030900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.459
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (310, 75)
INFO:tensorflow:loss = 0.1328125, step = 1031000 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.47
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1031100 (41.225 sec)
INFO:tensorflow:global_step/sec: 2.42574
INFO:tensorflow:examples/sec: 310.494
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1031100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1031100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1031100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1031100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (312, 0)
INFO:tensorflow:loss = 0.13476562, step = 1031200 (52.502 sec)
INFO:tensorflow:global_step/sec: 1.90466
INFO:tensorflow:examples/sec: 243.797
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (313, 48)
INFO:tensorflow:loss = 0.140625, step = 1031300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (314, 95)
INFO:tensorflow:loss = 0.13476562, step = 1031400 (41.222 sec)
INFO:tensorflow:global_step/sec: 2.42588
INFO:tensorflow:examples/sec: 310.512
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1031500 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.4566
INFO:tensorflow:examples/sec: 314.445
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (316, 43)
INFO:tensorflow:loss = 0.13769531, step = 1031600 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (317, 90)
INFO:tensorflow:loss = 0.13476562, step = 1031700 (41.145 sec)
INFO:tensorflow:global_step/sec: 2.43045
INFO:tensorflow:examples/sec: 311.098
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1031700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1031700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1031700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1031700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12792969, step = 1031800 (51.827 sec)
INFO:tensorflow:global_step/sec: 1.92951
INFO:tensorflow:examples/sec: 246.977
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (319, 11)
INFO:tensorflow:loss = 0.13378906, step = 1031900 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (320, 55)
INFO:tensorflow:loss = 0.140625, step = 1032000 (42.211 sec)
INFO:tensorflow:global_step/sec: 2.36903
INFO:tensorflow:examples/sec: 303.236
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13378906, step = 1032100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45661
INFO:tensorflow:examples/sec: 314.446
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (322, 3)
INFO:tensorflow:loss = 0.1328125, step = 1032200 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.463
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (323, 50)
INFO:tensorflow:loss = 0.1328125, step = 1032300 (41.229 sec)
INFO:tensorflow:global_step/sec: 2.4255
INFO:tensorflow:examples/sec: 310.464
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1032300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1032300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1032300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1032300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (324, 69)
INFO:tensorflow:loss = 0.13671875, step = 1032400 (52.284 sec)
INFO:tensorflow:global_step/sec: 1.91263
INFO:tensorflow:examples/sec: 244.816
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13671875, step = 1032500 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (326, 16)
INFO:tensorflow:loss = 0.140625, step = 1032600 (41.230 sec)
INFO:tensorflow:global_step/sec: 2.42545
INFO:tensorflow:examples/sec: 310.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (327, 64)
INFO:tensorflow:loss = 0.1328125, step = 1032700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.461
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1032800 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45682
INFO:tensorflow:examples/sec: 314.473
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (329, 10)
INFO:tensorflow:loss = 0.13085938, step = 1032900 (41.298 sec)
INFO:tensorflow:global_step/sec: 2.42143
INFO:tensorflow:examples/sec: 309.943
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1032900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1032900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1032900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1032900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (330, 31)
INFO:tensorflow:loss = 0.1328125, step = 1033000 (51.804 sec)
INFO:tensorflow:global_step/sec: 1.93034
INFO:tensorflow:examples/sec: 247.084
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (331, 79)
INFO:tensorflow:loss = 0.13671875, step = 1033100 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.461
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.11767578, step = 1033200 (41.259 sec)
INFO:tensorflow:global_step/sec: 2.42373
INFO:tensorflow:examples/sec: 310.238
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (333, 26)
INFO:tensorflow:loss = 0.1328125, step = 1033300 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (334, 74)
INFO:tensorflow:loss = 0.12792969, step = 1033400 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45684
INFO:tensorflow:examples/sec: 314.475
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1328125, step = 1033500 (41.231 sec)
INFO:tensorflow:global_step/sec: 2.42536
INFO:tensorflow:examples/sec: 310.446
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1033500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1033500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1033500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1033500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (336, 0)
INFO:tensorflow:loss = 0.13085938, step = 1033600 (52.224 sec)
INFO:tensorflow:global_step/sec: 1.91481
INFO:tensorflow:examples/sec: 245.096
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (337, 48)
INFO:tensorflow:loss = 0.140625, step = 1033700 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45683
INFO:tensorflow:examples/sec: 314.475
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (338, 95)
INFO:tensorflow:loss = 0.13671875, step = 1033800 (41.113 sec)
INFO:tensorflow:global_step/sec: 2.43231
INFO:tensorflow:examples/sec: 311.336
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1033900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (340, 43)
INFO:tensorflow:loss = 0.13671875, step = 1034000 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (341, 89)
INFO:tensorflow:loss = 0.13671875, step = 1034100 (41.362 sec)
INFO:tensorflow:global_step/sec: 2.41769
INFO:tensorflow:examples/sec: 309.464
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1034100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1034100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1034100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1034100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13378906, step = 1034200 (52.291 sec)
INFO:tensorflow:global_step/sec: 1.91238
INFO:tensorflow:examples/sec: 244.785
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (343, 8)
INFO:tensorflow:loss = 0.13574219, step = 1034300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (344, 54)
INFO:tensorflow:loss = 0.13085938, step = 1034400 (41.289 sec)
INFO:tensorflow:global_step/sec: 2.42197
INFO:tensorflow:examples/sec: 310.012
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12988281, step = 1034500 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (346, 2)
INFO:tensorflow:loss = 0.125, step = 1034600 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.47
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (347, 49)
INFO:tensorflow:loss = 0.12792969, step = 1034700 (41.272 sec)
INFO:tensorflow:global_step/sec: 2.42292
INFO:tensorflow:examples/sec: 310.134
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1034700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1034700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1034700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1034700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (348, 67)
INFO:tensorflow:loss = 0.1328125, step = 1034800 (52.743 sec)
INFO:tensorflow:global_step/sec: 1.896
INFO:tensorflow:examples/sec: 242.688
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13574219, step = 1034900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.463
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (350, 14)
INFO:tensorflow:loss = 0.140625, step = 1035000 (41.168 sec)
INFO:tensorflow:global_step/sec: 2.42909
INFO:tensorflow:examples/sec: 310.924
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (351, 62)
INFO:tensorflow:loss = 0.14355469, step = 1035100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45662
INFO:tensorflow:examples/sec: 314.447
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1035200 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45687
INFO:tensorflow:examples/sec: 314.48
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (353, 9)
INFO:tensorflow:loss = 0.13671875, step = 1035300 (41.141 sec)
INFO:tensorflow:global_step/sec: 2.43063
INFO:tensorflow:examples/sec: 311.121
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1035300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1035300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1035300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1035300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (354, 30)
INFO:tensorflow:loss = 0.12792969, step = 1035400 (51.797 sec)
INFO:tensorflow:global_step/sec: 1.9306
INFO:tensorflow:examples/sec: 247.117
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (355, 78)
INFO:tensorflow:loss = 0.13085938, step = 1035500 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45685
INFO:tensorflow:examples/sec: 314.477
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13085938, step = 1035600 (41.321 sec)
INFO:tensorflow:global_step/sec: 2.42002
INFO:tensorflow:examples/sec: 309.763
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (357, 24)
INFO:tensorflow:loss = 0.13085938, step = 1035700 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (358, 72)
INFO:tensorflow:loss = 0.125, step = 1035800 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45681
INFO:tensorflow:examples/sec: 314.472
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13574219, step = 1035900 (41.206 sec)
INFO:tensorflow:global_step/sec: 2.42685
INFO:tensorflow:examples/sec: 310.636
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1035900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1035900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1035900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1035900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (360, 0)
INFO:tensorflow:loss = 0.15234375, step = 1036000 (52.753 sec)
INFO:tensorflow:global_step/sec: 1.89563
INFO:tensorflow:examples/sec: 242.641
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (361, 48)
INFO:tensorflow:loss = 0.125, step = 1036100 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45685
INFO:tensorflow:examples/sec: 314.476
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (362, 94)
INFO:tensorflow:loss = 0.12792969, step = 1036200 (41.591 sec)
INFO:tensorflow:global_step/sec: 2.40434
INFO:tensorflow:examples/sec: 307.755
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1328125, step = 1036300 (40.708 sec)
INFO:tensorflow:global_step/sec: 2.45652
INFO:tensorflow:examples/sec: 314.435
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (364, 42)
INFO:tensorflow:loss = 0.13085938, step = 1036400 (40.707 sec)
INFO:tensorflow:global_step/sec: 2.45658
INFO:tensorflow:examples/sec: 314.443
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (365, 88)
INFO:tensorflow:loss = 0.13867188, step = 1036500 (41.442 sec)
INFO:tensorflow:global_step/sec: 2.41307
INFO:tensorflow:examples/sec: 308.873
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1036500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1036500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1036500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1036500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1036600 (54.065 sec)
INFO:tensorflow:global_step/sec: 1.84962
INFO:tensorflow:examples/sec: 236.752
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (367, 3)
INFO:tensorflow:loss = 0.14257812, step = 1036700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (368, 50)
INFO:tensorflow:loss = 0.140625, step = 1036800 (41.125 sec)
INFO:tensorflow:global_step/sec: 2.43159
INFO:tensorflow:examples/sec: 311.244
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (369, 98)
INFO:tensorflow:loss = 0.13867188, step = 1036900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1037000 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45686
INFO:tensorflow:examples/sec: 314.478
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (371, 44)
INFO:tensorflow:loss = 0.12695312, step = 1037100 (41.308 sec)
INFO:tensorflow:global_step/sec: 2.42081
INFO:tensorflow:examples/sec: 309.864
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1037100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1037100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1037100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1037100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (372, 64)
INFO:tensorflow:loss = 0.13085938, step = 1037200 (51.915 sec)
INFO:tensorflow:global_step/sec: 1.92621
INFO:tensorflow:examples/sec: 246.555
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13574219, step = 1037300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (374, 11)
INFO:tensorflow:loss = 0.13671875, step = 1037400 (41.106 sec)
INFO:tensorflow:global_step/sec: 2.43271
INFO:tensorflow:examples/sec: 311.387
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (375, 59)
INFO:tensorflow:loss = 0.14648438, step = 1037500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14355469, step = 1037600 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.4568
INFO:tensorflow:examples/sec: 314.47
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (377, 6)
INFO:tensorflow:loss = 0.13183594, step = 1037700 (41.254 sec)
INFO:tensorflow:global_step/sec: 2.42402
INFO:tensorflow:examples/sec: 310.274
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1037700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1037700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1037700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1037700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (378, 23)
INFO:tensorflow:loss = 0.13183594, step = 1037800 (53.377 sec)
INFO:tensorflow:global_step/sec: 1.87346
INFO:tensorflow:examples/sec: 239.802
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (379, 71)
INFO:tensorflow:loss = 0.13867188, step = 1037900 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45664
INFO:tensorflow:examples/sec: 314.45
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.140625, step = 1038000 (41.204 sec)
INFO:tensorflow:global_step/sec: 2.42699
INFO:tensorflow:examples/sec: 310.655
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (381, 18)
INFO:tensorflow:loss = 0.12792969, step = 1038100 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (382, 66)
INFO:tensorflow:loss = 0.13671875, step = 1038200 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13574219, step = 1038300 (41.157 sec)
INFO:tensorflow:global_step/sec: 2.42975
INFO:tensorflow:examples/sec: 311.009
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1038300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1038300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1038300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1038300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (384, 0)
INFO:tensorflow:loss = 0.14453125, step = 1038400 (52.181 sec)
INFO:tensorflow:global_step/sec: 1.9164
INFO:tensorflow:examples/sec: 245.3
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (385, 48)
INFO:tensorflow:loss = 0.13574219, step = 1038500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (386, 94)
INFO:tensorflow:loss = 0.13476562, step = 1038600 (41.500 sec)
INFO:tensorflow:global_step/sec: 2.40962
INFO:tensorflow:examples/sec: 308.431
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1328125, step = 1038700 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.451
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (388, 42)
INFO:tensorflow:loss = 0.13476562, step = 1038800 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45681
INFO:tensorflow:examples/sec: 314.472
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (389, 89)
INFO:tensorflow:loss = 0.140625, step = 1038900 (41.202 sec)
INFO:tensorflow:global_step/sec: 2.42706
INFO:tensorflow:examples/sec: 310.664
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1038900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1038900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1038900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1038900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13085938, step = 1039000 (55.931 sec)
INFO:tensorflow:global_step/sec: 1.7879
INFO:tensorflow:examples/sec: 228.851
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (391, 0)
INFO:tensorflow:loss = 0.1328125, step = 1039100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.455
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (392, 47)
INFO:tensorflow:loss = 0.14355469, step = 1039200 (41.153 sec)
INFO:tensorflow:global_step/sec: 2.42996
INFO:tensorflow:examples/sec: 311.035
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (393, 95)
INFO:tensorflow:loss = 0.14453125, step = 1039300 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45664
INFO:tensorflow:examples/sec: 314.45
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1039400 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.451
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (395, 42)
INFO:tensorflow:loss = 0.14648438, step = 1039500 (41.111 sec)
INFO:tensorflow:global_step/sec: 2.43245
INFO:tensorflow:examples/sec: 311.354
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1039500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1039500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1039500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1039500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (396, 65)
INFO:tensorflow:loss = 0.13476562, step = 1039600 (51.025 sec)
INFO:tensorflow:global_step/sec: 1.95985
INFO:tensorflow:examples/sec: 250.861
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13574219, step = 1039700 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45682
INFO:tensorflow:examples/sec: 314.473
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (398, 12)
INFO:tensorflow:loss = 0.13671875, step = 1039800 (41.158 sec)
INFO:tensorflow:global_step/sec: 2.42967
INFO:tensorflow:examples/sec: 310.998
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (399, 60)
INFO:tensorflow:loss = 0.13574219, step = 1039900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1040000 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45683
INFO:tensorflow:examples/sec: 314.474
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (401, 7)
INFO:tensorflow:loss = 0.13671875, step = 1040100 (41.117 sec)
INFO:tensorflow:global_step/sec: 2.43205
INFO:tensorflow:examples/sec: 311.303
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1040100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1040100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1040100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1040100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (402, 17)
INFO:tensorflow:loss = 0.13574219, step = 1040200 (56.186 sec)
INFO:tensorflow:global_step/sec: 1.77981
INFO:tensorflow:examples/sec: 227.816
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (403, 65)
INFO:tensorflow:loss = 0.1328125, step = 1040300 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45683
INFO:tensorflow:examples/sec: 314.475
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13671875, step = 1040400 (41.149 sec)
INFO:tensorflow:global_step/sec: 2.43018
INFO:tensorflow:examples/sec: 311.062
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (405, 12)
INFO:tensorflow:loss = 0.13085938, step = 1040500 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.459
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (406, 60)
INFO:tensorflow:loss = 0.140625, step = 1040600 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.468
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12988281, step = 1040700 (41.334 sec)
INFO:tensorflow:global_step/sec: 2.41932
INFO:tensorflow:examples/sec: 309.673
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1040700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1040700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1040700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1040700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (408, 0)
INFO:tensorflow:loss = 0.13085938, step = 1040800 (51.500 sec)
INFO:tensorflow:global_step/sec: 1.94174
INFO:tensorflow:examples/sec: 248.543
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (409, 48)
INFO:tensorflow:loss = 0.12695312, step = 1040900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (410, 95)
INFO:tensorflow:loss = 0.13085938, step = 1041000 (41.119 sec)
INFO:tensorflow:global_step/sec: 2.43198
INFO:tensorflow:examples/sec: 311.294
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1041100 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (412, 43)
INFO:tensorflow:loss = 0.13476562, step = 1041200 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.47
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (413, 90)
INFO:tensorflow:loss = 0.140625, step = 1041300 (41.249 sec)
INFO:tensorflow:global_step/sec: 2.4243
INFO:tensorflow:examples/sec: 310.31
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1041300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1041300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1041300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1041300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1041400 (51.085 sec)
INFO:tensorflow:global_step/sec: 1.95753
INFO:tensorflow:examples/sec: 250.564
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (415, 12)
INFO:tensorflow:loss = 0.14257812, step = 1041500 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45685
INFO:tensorflow:examples/sec: 314.477
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (416, 59)
INFO:tensorflow:loss = 0.14355469, step = 1041600 (41.230 sec)
INFO:tensorflow:global_step/sec: 2.42542
INFO:tensorflow:examples/sec: 310.453
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.140625, step = 1041700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.459
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (418, 7)
INFO:tensorflow:loss = 0.14453125, step = 1041800 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45688
INFO:tensorflow:examples/sec: 314.481
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (419, 54)
INFO:tensorflow:loss = 0.14453125, step = 1041900 (41.163 sec)
INFO:tensorflow:global_step/sec: 2.42936
INFO:tensorflow:examples/sec: 310.958
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1041900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1041900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1041900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1041900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (420, 74)
INFO:tensorflow:loss = 0.14648438, step = 1042000 (51.895 sec)
INFO:tensorflow:global_step/sec: 1.92697
INFO:tensorflow:examples/sec: 246.652
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14257812, step = 1042100 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (422, 20)
INFO:tensorflow:loss = 0.13867188, step = 1042200 (41.672 sec)
INFO:tensorflow:global_step/sec: 2.39969
INFO:tensorflow:examples/sec: 307.16
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (423, 68)
INFO:tensorflow:loss = 0.14746094, step = 1042300 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14355469, step = 1042400 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45684
INFO:tensorflow:examples/sec: 314.476
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (425, 15)
INFO:tensorflow:loss = 0.140625, step = 1042500 (41.210 sec)
INFO:tensorflow:global_step/sec: 2.42661
INFO:tensorflow:examples/sec: 310.607
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1042500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1042500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1042500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1042500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (426, 34)
INFO:tensorflow:loss = 0.1328125, step = 1042600 (52.310 sec)
INFO:tensorflow:global_step/sec: 1.91167
INFO:tensorflow:examples/sec: 244.694
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (427, 82)
INFO:tensorflow:loss = 0.13085938, step = 1042700 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.4568
INFO:tensorflow:examples/sec: 314.471
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1328125, step = 1042800 (41.376 sec)
INFO:tensorflow:global_step/sec: 2.41687
INFO:tensorflow:examples/sec: 309.359
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (429, 28)
INFO:tensorflow:loss = 0.1328125, step = 1042900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (430, 76)
INFO:tensorflow:loss = 0.12890625, step = 1043000 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.140625, step = 1043100 (41.246 sec)
INFO:tensorflow:global_step/sec: 2.42451
INFO:tensorflow:examples/sec: 310.337
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1043100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1043100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1043100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1043100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (432, 0)
INFO:tensorflow:loss = 0.140625, step = 1043200 (55.968 sec)
INFO:tensorflow:global_step/sec: 1.78675
INFO:tensorflow:examples/sec: 228.704
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (433, 48)
INFO:tensorflow:loss = 0.14453125, step = 1043300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (434, 93)
INFO:tensorflow:loss = 0.14160156, step = 1043400 (41.734 sec)
INFO:tensorflow:global_step/sec: 2.39611
INFO:tensorflow:examples/sec: 306.702
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14648438, step = 1043500 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (436, 41)
INFO:tensorflow:loss = 0.13671875, step = 1043600 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45682
INFO:tensorflow:examples/sec: 314.473
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (437, 88)
INFO:tensorflow:loss = 0.14355469, step = 1043700 (41.181 sec)
INFO:tensorflow:global_step/sec: 2.42834
INFO:tensorflow:examples/sec: 310.827
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1043700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1043700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1043700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1043700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1043800 (50.856 sec)
INFO:tensorflow:global_step/sec: 1.96634
INFO:tensorflow:examples/sec: 251.692
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (439, 11)
INFO:tensorflow:loss = 0.14453125, step = 1043900 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45688
INFO:tensorflow:examples/sec: 314.481
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (440, 58)
INFO:tensorflow:loss = 0.14355469, step = 1044000 (41.176 sec)
INFO:tensorflow:global_step/sec: 2.42861
INFO:tensorflow:examples/sec: 310.862
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14355469, step = 1044100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (442, 6)
INFO:tensorflow:loss = 0.13769531, step = 1044200 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (443, 50)
INFO:tensorflow:loss = 0.13867188, step = 1044300 (42.476 sec)
INFO:tensorflow:global_step/sec: 2.35429
INFO:tensorflow:examples/sec: 301.349
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1044300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1044300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1044300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1044300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (444, 40)
INFO:tensorflow:loss = 0.13671875, step = 1044400 (64.167 sec)
INFO:tensorflow:global_step/sec: 1.55845
INFO:tensorflow:examples/sec: 199.482
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (445, 88)
INFO:tensorflow:loss = 0.14257812, step = 1044500 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45666
INFO:tensorflow:examples/sec: 314.453
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13574219, step = 1044600 (41.279 sec)
INFO:tensorflow:global_step/sec: 2.42254
INFO:tensorflow:examples/sec: 310.085
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (447, 34)
INFO:tensorflow:loss = 0.14160156, step = 1044700 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.4568
INFO:tensorflow:examples/sec: 314.47
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (448, 82)
INFO:tensorflow:loss = 0.13867188, step = 1044800 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.14453125, step = 1044900 (41.280 sec)
INFO:tensorflow:global_step/sec: 2.4225
INFO:tensorflow:examples/sec: 310.08
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1044900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1044900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1044900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1044900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (450, 0)
INFO:tensorflow:loss = 0.140625, step = 1045000 (52.428 sec)
INFO:tensorflow:global_step/sec: 1.90736
INFO:tensorflow:examples/sec: 244.142
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (451, 48)
INFO:tensorflow:loss = 0.140625, step = 1045100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.451
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (452, 95)
INFO:tensorflow:loss = 0.13183594, step = 1045200 (41.088 sec)
INFO:tensorflow:global_step/sec: 2.43381
INFO:tensorflow:examples/sec: 311.528
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1045300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.468
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (454, 43)
INFO:tensorflow:loss = 0.13671875, step = 1045400 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.468
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (455, 89)
INFO:tensorflow:loss = 0.14648438, step = 1045500 (41.297 sec)
INFO:tensorflow:global_step/sec: 2.42147
INFO:tensorflow:examples/sec: 309.949
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1045500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1045500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1045500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1045500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13867188, step = 1045600 (52.073 sec)
INFO:tensorflow:global_step/sec: 1.9204
INFO:tensorflow:examples/sec: 245.811
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (457, 9)
INFO:tensorflow:loss = 0.13867188, step = 1045700 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.46
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (458, 56)
INFO:tensorflow:loss = 0.13964844, step = 1045800 (41.210 sec)
INFO:tensorflow:global_step/sec: 2.42661
INFO:tensorflow:examples/sec: 310.606
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13769531, step = 1045900 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (460, 4)
INFO:tensorflow:loss = 0.13085938, step = 1046000 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45689
INFO:tensorflow:examples/sec: 314.481
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (461, 51)
INFO:tensorflow:loss = 0.13671875, step = 1046100 (41.218 sec)
INFO:tensorflow:global_step/sec: 2.42614
INFO:tensorflow:examples/sec: 310.546
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1046100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1046100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1046100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1046100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (462, 70)
INFO:tensorflow:loss = 0.13671875, step = 1046200 (52.645 sec)
INFO:tensorflow:global_step/sec: 1.89951
INFO:tensorflow:examples/sec: 243.137
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13085938, step = 1046300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (464, 17)
INFO:tensorflow:loss = 0.14648438, step = 1046400 (41.195 sec)
INFO:tensorflow:global_step/sec: 2.42751
INFO:tensorflow:examples/sec: 310.721
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (465, 65)
INFO:tensorflow:loss = 0.14355469, step = 1046500 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45661
INFO:tensorflow:examples/sec: 314.447
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13671875, step = 1046600 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45672
INFO:tensorflow:examples/sec: 314.461
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (467, 12)
INFO:tensorflow:loss = 0.13867188, step = 1046700 (41.194 sec)
INFO:tensorflow:global_step/sec: 2.42756
INFO:tensorflow:examples/sec: 310.728
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1046700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1046700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1046700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1046700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (468, 32)
INFO:tensorflow:loss = 0.140625, step = 1046800 (51.914 sec)
INFO:tensorflow:global_step/sec: 1.92629
INFO:tensorflow:examples/sec: 246.565
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (469, 80)
INFO:tensorflow:loss = 0.14355469, step = 1046900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13964844, step = 1047000 (41.144 sec)
INFO:tensorflow:global_step/sec: 2.4305
INFO:tensorflow:examples/sec: 311.105
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (471, 27)
INFO:tensorflow:loss = 0.14257812, step = 1047100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.456
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (472, 75)
INFO:tensorflow:loss = 0.13671875, step = 1047200 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.47
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.123535156, step = 1047300 (45.897 sec)
INFO:tensorflow:global_step/sec: 2.17876
INFO:tensorflow:examples/sec: 278.881
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1047300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1047300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1047300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1047300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (474, 0)
INFO:tensorflow:loss = 0.1328125, step = 1047400 (52.063 sec)
INFO:tensorflow:global_step/sec: 1.92074
INFO:tensorflow:examples/sec: 245.855
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (475, 48)
INFO:tensorflow:loss = 0.14453125, step = 1047500 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45684
INFO:tensorflow:examples/sec: 314.476
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (476, 95)
INFO:tensorflow:loss = 0.12890625, step = 1047600 (41.147 sec)
INFO:tensorflow:global_step/sec: 2.43029
INFO:tensorflow:examples/sec: 311.077
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12792969, step = 1047700 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (478, 43)
INFO:tensorflow:loss = 0.13867188, step = 1047800 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (479, 90)
INFO:tensorflow:loss = 0.13476562, step = 1047900 (41.028 sec)
INFO:tensorflow:global_step/sec: 2.43734
INFO:tensorflow:examples/sec: 311.979
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1047900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1047900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1047900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1047900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1048000 (51.267 sec)
INFO:tensorflow:global_step/sec: 1.95058
INFO:tensorflow:examples/sec: 249.674
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (481, 12)
INFO:tensorflow:loss = 0.13574219, step = 1048100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (482, 59)
INFO:tensorflow:loss = 0.13867188, step = 1048200 (41.127 sec)
INFO:tensorflow:global_step/sec: 2.43149
INFO:tensorflow:examples/sec: 311.23
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1048300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.468
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (484, 7)
INFO:tensorflow:loss = 0.12792969, step = 1048400 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.4568
INFO:tensorflow:examples/sec: 314.471
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (485, 54)
INFO:tensorflow:loss = 0.13183594, step = 1048500 (41.224 sec)
INFO:tensorflow:global_step/sec: 2.42576
INFO:tensorflow:examples/sec: 310.498
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1048500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1048500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1048500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1048500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (486, 75)
INFO:tensorflow:loss = 0.13476562, step = 1048600 (51.774 sec)
INFO:tensorflow:global_step/sec: 1.93146
INFO:tensorflow:examples/sec: 247.227
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12695312, step = 1048700 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45663
INFO:tensorflow:examples/sec: 314.448
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (488, 22)
INFO:tensorflow:loss = 0.13476562, step = 1048800 (41.206 sec)
INFO:tensorflow:global_step/sec: 2.42687
INFO:tensorflow:examples/sec: 310.639
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (489, 70)
INFO:tensorflow:loss = 0.12988281, step = 1048900 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.451
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1049000 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.468
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (491, 17)
INFO:tensorflow:loss = 0.12988281, step = 1049100 (41.032 sec)
INFO:tensorflow:global_step/sec: 2.43709
INFO:tensorflow:examples/sec: 311.947
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1049100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1049100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1049100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1049100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (492, 38)
INFO:tensorflow:loss = 0.13085938, step = 1049200 (51.596 sec)
INFO:tensorflow:global_step/sec: 1.93817
INFO:tensorflow:examples/sec: 248.086
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (493, 86)
INFO:tensorflow:loss = 0.13085938, step = 1049300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.463
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13085938, step = 1049400 (41.302 sec)
INFO:tensorflow:global_step/sec: 2.42125
INFO:tensorflow:examples/sec: 309.92
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (495, 32)
INFO:tensorflow:loss = 0.13085938, step = 1049500 (40.707 sec)
INFO:tensorflow:global_step/sec: 2.45655
INFO:tensorflow:examples/sec: 314.439
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (496, 80)
INFO:tensorflow:loss = 0.125, step = 1049600 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13671875, step = 1049700 (41.203 sec)
INFO:tensorflow:global_step/sec: 2.42701
INFO:tensorflow:examples/sec: 310.657
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1049700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1049700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1049700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1049700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (498, 0)
INFO:tensorflow:loss = 0.13183594, step = 1049800 (52.280 sec)
INFO:tensorflow:global_step/sec: 1.91278
INFO:tensorflow:examples/sec: 244.836
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (499, 48)
INFO:tensorflow:loss = 0.125, step = 1049900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (500, 94)
INFO:tensorflow:loss = 0.13671875, step = 1050000 (41.324 sec)
INFO:tensorflow:global_step/sec: 2.41989
INFO:tensorflow:examples/sec: 309.746
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12890625, step = 1050100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.456
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (502, 42)
INFO:tensorflow:loss = 0.13183594, step = 1050200 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (503, 89)
INFO:tensorflow:loss = 0.13476562, step = 1050300 (41.161 sec)
INFO:tensorflow:global_step/sec: 2.42947
INFO:tensorflow:examples/sec: 310.972
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1050300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1050300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1050300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1050300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13085938, step = 1050400 (50.655 sec)
INFO:tensorflow:global_step/sec: 1.97416
INFO:tensorflow:examples/sec: 252.693
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (505, 12)
INFO:tensorflow:loss = 0.14453125, step = 1050500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (506, 59)
INFO:tensorflow:loss = 0.13085938, step = 1050600 (41.152 sec)
INFO:tensorflow:global_step/sec: 2.42999
INFO:tensorflow:examples/sec: 311.038
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12988281, step = 1050700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45671
INFO:tensorflow:examples/sec: 314.459
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (508, 7)
INFO:tensorflow:loss = 0.13476562, step = 1050800 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (509, 54)
INFO:tensorflow:loss = 0.13183594, step = 1050900 (41.069 sec)
INFO:tensorflow:global_step/sec: 2.43491
INFO:tensorflow:examples/sec: 311.668
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1050900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1050900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1050900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1050900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (510, 78)
INFO:tensorflow:loss = 0.12402344, step = 1051000 (50.345 sec)
INFO:tensorflow:global_step/sec: 1.98629
INFO:tensorflow:examples/sec: 254.245
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1051100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.456
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (512, 25)
INFO:tensorflow:loss = 0.13085938, step = 1051200 (41.066 sec)
INFO:tensorflow:global_step/sec: 2.43513
INFO:tensorflow:examples/sec: 311.696
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (513, 73)
INFO:tensorflow:loss = 0.12695312, step = 1051300 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45664
INFO:tensorflow:examples/sec: 314.45
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1328125, step = 1051400 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.456
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (515, 20)
INFO:tensorflow:loss = 0.12695312, step = 1051500 (41.161 sec)
INFO:tensorflow:global_step/sec: 2.42946
INFO:tensorflow:examples/sec: 310.971
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1051500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1051500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1051500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1051500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (516, 40)
INFO:tensorflow:loss = 0.12695312, step = 1051600 (52.018 sec)
INFO:tensorflow:global_step/sec: 1.92241
INFO:tensorflow:examples/sec: 246.069
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (517, 88)
INFO:tensorflow:loss = 0.13378906, step = 1051700 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12890625, step = 1051800 (41.203 sec)
INFO:tensorflow:global_step/sec: 2.42703
INFO:tensorflow:examples/sec: 310.659
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (519, 35)
INFO:tensorflow:loss = 0.13085938, step = 1051900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (520, 83)
INFO:tensorflow:loss = 0.13085938, step = 1052000 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.455
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12890625, step = 1052100 (41.238 sec)
INFO:tensorflow:global_step/sec: 2.425
INFO:tensorflow:examples/sec: 310.4
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1052100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1052100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1052100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1052100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (522, 1)
INFO:tensorflow:loss = 0.13574219, step = 1052200 (52.265 sec)
INFO:tensorflow:global_step/sec: 1.91329
INFO:tensorflow:examples/sec: 244.901
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (523, 49)
INFO:tensorflow:loss = 0.12890625, step = 1052300 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (524, 96)
INFO:tensorflow:loss = 0.1328125, step = 1052400 (41.184 sec)
INFO:tensorflow:global_step/sec: 2.42812
INFO:tensorflow:examples/sec: 310.799
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12207031, step = 1052500 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.458
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (526, 44)
INFO:tensorflow:loss = 0.13378906, step = 1052600 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45681
INFO:tensorflow:examples/sec: 314.471
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (527, 90)
INFO:tensorflow:loss = 0.13476562, step = 1052700 (41.457 sec)
INFO:tensorflow:global_step/sec: 2.41215
INFO:tensorflow:examples/sec: 308.755
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1052700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1052700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1052700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1052700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.125, step = 1052800 (51.837 sec)
INFO:tensorflow:global_step/sec: 1.92912
INFO:tensorflow:examples/sec: 246.928
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (529, 11)
INFO:tensorflow:loss = 0.11669922, step = 1052900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (530, 58)
INFO:tensorflow:loss = 0.12792969, step = 1053000 (41.240 sec)
INFO:tensorflow:global_step/sec: 2.42481
INFO:tensorflow:examples/sec: 310.376
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.15820312, step = 1053100 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (532, 6)
INFO:tensorflow:loss = 0.125, step = 1053200 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (533, 53)
INFO:tensorflow:loss = 0.13671875, step = 1053300 (41.247 sec)
INFO:tensorflow:global_step/sec: 2.42443
INFO:tensorflow:examples/sec: 310.327
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1053300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1053300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1053300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1053300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (534, 74)
INFO:tensorflow:loss = 0.12988281, step = 1053400 (51.546 sec)
INFO:tensorflow:global_step/sec: 1.94
INFO:tensorflow:examples/sec: 248.32
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13085938, step = 1053500 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45674
INFO:tensorflow:examples/sec: 314.463
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (536, 20)
INFO:tensorflow:loss = 0.13085938, step = 1053600 (41.414 sec)
INFO:tensorflow:global_step/sec: 2.41462
INFO:tensorflow:examples/sec: 309.071
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (537, 68)
INFO:tensorflow:loss = 0.12890625, step = 1053700 (40.707 sec)
INFO:tensorflow:global_step/sec: 2.45658
INFO:tensorflow:examples/sec: 314.443
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12890625, step = 1053800 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45678
INFO:tensorflow:examples/sec: 314.468
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (539, 14)
INFO:tensorflow:loss = 0.12792969, step = 1053900 (41.402 sec)
INFO:tensorflow:global_step/sec: 2.41534
INFO:tensorflow:examples/sec: 309.164
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1053900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1053900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1053900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1053900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (540, 24)
INFO:tensorflow:loss = 0.12109375, step = 1054000 (55.966 sec)
INFO:tensorflow:global_step/sec: 1.7868
INFO:tensorflow:examples/sec: 228.711
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (541, 72)
INFO:tensorflow:loss = 0.13867188, step = 1054100 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.4567
INFO:tensorflow:examples/sec: 314.457
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1054200 (41.183 sec)
INFO:tensorflow:global_step/sec: 2.42822
INFO:tensorflow:examples/sec: 310.812
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (543, 19)
INFO:tensorflow:loss = 0.1328125, step = 1054300 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.456
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (544, 67)
INFO:tensorflow:loss = 0.12597656, step = 1054400 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45675
INFO:tensorflow:examples/sec: 314.464
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.1328125, step = 1054500 (41.366 sec)
INFO:tensorflow:global_step/sec: 2.41745
INFO:tensorflow:examples/sec: 309.433
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1054500...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1054500 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1054500 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1054500...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (546, 0)
INFO:tensorflow:loss = 0.13964844, step = 1054600 (52.430 sec)
INFO:tensorflow:global_step/sec: 1.9073
INFO:tensorflow:examples/sec: 244.134
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (547, 48)
INFO:tensorflow:loss = 0.140625, step = 1054700 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45668
INFO:tensorflow:examples/sec: 314.455
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (548, 95)
INFO:tensorflow:loss = 0.13867188, step = 1054800 (41.233 sec)
INFO:tensorflow:global_step/sec: 2.42523
INFO:tensorflow:examples/sec: 310.43
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1054900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45677
INFO:tensorflow:examples/sec: 314.467
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (550, 43)
INFO:tensorflow:loss = 0.13476562, step = 1055000 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45683
INFO:tensorflow:examples/sec: 314.474
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (551, 90)
INFO:tensorflow:loss = 0.13671875, step = 1055100 (41.074 sec)
INFO:tensorflow:global_step/sec: 2.43459
INFO:tensorflow:examples/sec: 311.628
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1055100...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1055100 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1055100 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1055100...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13769531, step = 1055200 (51.946 sec)
INFO:tensorflow:global_step/sec: 1.92508
INFO:tensorflow:examples/sec: 246.41
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (553, 10)
INFO:tensorflow:loss = 0.13476562, step = 1055300 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (554, 57)
INFO:tensorflow:loss = 0.13671875, step = 1055400 (41.213 sec)
INFO:tensorflow:global_step/sec: 2.42643
INFO:tensorflow:examples/sec: 310.584
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12695312, step = 1055500 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45664
INFO:tensorflow:examples/sec: 314.45
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (556, 5)
INFO:tensorflow:loss = 0.13671875, step = 1055600 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.466
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (557, 52)
INFO:tensorflow:loss = 0.13476562, step = 1055700 (41.225 sec)
INFO:tensorflow:global_step/sec: 2.42572
INFO:tensorflow:examples/sec: 310.492
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1055700...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1055700 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1055700 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1055700...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (558, 74)
INFO:tensorflow:loss = 0.13867188, step = 1055800 (51.348 sec)
INFO:tensorflow:global_step/sec: 1.9475
INFO:tensorflow:examples/sec: 249.28
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12890625, step = 1055900 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45673
INFO:tensorflow:examples/sec: 314.462
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (560, 21)
INFO:tensorflow:loss = 0.13476562, step = 1056000 (41.075 sec)
INFO:tensorflow:global_step/sec: 2.43458
INFO:tensorflow:examples/sec: 311.626
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (561, 69)
INFO:tensorflow:loss = 0.12890625, step = 1056100 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45665
INFO:tensorflow:examples/sec: 314.452
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.140625, step = 1056200 (40.704 sec)
INFO:tensorflow:global_step/sec: 2.45676
INFO:tensorflow:examples/sec: 314.465
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (563, 16)
INFO:tensorflow:loss = 0.1328125, step = 1056300 (41.200 sec)
INFO:tensorflow:global_step/sec: 2.42719
INFO:tensorflow:examples/sec: 310.68
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1056300...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1056300 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1056300 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1056300...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (564, 35)
INFO:tensorflow:loss = 0.13671875, step = 1056400 (52.297 sec)
INFO:tensorflow:global_step/sec: 1.91218
INFO:tensorflow:examples/sec: 244.759
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (565, 83)
INFO:tensorflow:loss = 0.14160156, step = 1056500 (40.705 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.454
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1056600 (41.118 sec)
INFO:tensorflow:global_step/sec: 2.432
INFO:tensorflow:examples/sec: 311.296
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (567, 30)
INFO:tensorflow:loss = 0.13769531, step = 1056700 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45667
INFO:tensorflow:examples/sec: 314.453
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (568, 78)
INFO:tensorflow:loss = 0.140625, step = 1056800 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45687
INFO:tensorflow:examples/sec: 314.48
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13574219, step = 1056900 (41.270 sec)
INFO:tensorflow:global_step/sec: 2.42305
INFO:tensorflow:examples/sec: 310.151
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1056900...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1056900 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1056900 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1056900...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (570, 0)
INFO:tensorflow:loss = 0.12792969, step = 1057000 (51.465 sec)
INFO:tensorflow:global_step/sec: 1.94309
INFO:tensorflow:examples/sec: 248.716
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (571, 48)
INFO:tensorflow:loss = 0.12597656, step = 1057100 (40.702 sec)
INFO:tensorflow:global_step/sec: 2.45688
INFO:tensorflow:examples/sec: 314.481
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (572, 95)
INFO:tensorflow:loss = 0.13671875, step = 1057200 (41.193 sec)
INFO:tensorflow:global_step/sec: 2.42756
INFO:tensorflow:examples/sec: 310.728
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.12890625, step = 1057300 (40.706 sec)
INFO:tensorflow:global_step/sec: 2.45669
INFO:tensorflow:examples/sec: 314.456
INFO:tensorflow:Enqueue next (100) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (100) batch(es) of data from outfeed.
INFO:tensorflow:Outfeed finished for iteration (574, 43)
INFO:tensorflow:loss = 0.13769531, step = 1057400 (40.703 sec)
INFO:tensorflow:global_step/sec: 2.45679
INFO:tensorflow:examples/sec: 314.469
INFO:tensorflow:Enqueue next (10) batch(es) of data to infeed.
INFO:tensorflow:Dequeue next (10) batch(es) of data from outfeed.
INFO:tensorflow:loss = 0.13476562, step = 1057410 (4.576 sec)
INFO:tensorflow:global_step/sec: 2.18553
INFO:tensorflow:examples/sec: 279.747
INFO:tensorflow:Calling checkpoint listeners before saving checkpoint 1057410...
INFO:tensorflow:Before Save.
INFO:tensorflow:About to write a checkpoint
INFO:tensorflow:Saving checkpoints for 1057410 into gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt.
INFO:tensorflow:gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1057410 is not in all_model_checkpoint_paths. Manually adding it.
INFO:tensorflow:Calling checkpoint listeners after saving checkpoint 1057410...
INFO:tensorflow:Done writing checkpoint.
INFO:tensorflow:Stop infeed thread controller
INFO:tensorflow:Shutting down InfeedController thread.
INFO:tensorflow:InfeedController received shutdown signal, stopping.
INFO:tensorflow:Infeed thread finished, shutting down.
INFO:tensorflow:infeed marked as finished
INFO:tensorflow:Stop output thread controller
INFO:tensorflow:Shutting down OutfeedController thread.
INFO:tensorflow:OutfeedController received shutdown signal, stopping.
INFO:tensorflow:Outfeed thread finished, shutting down.
INFO:tensorflow:outfeed marked as finished
INFO:tensorflow:Shutdown TPU system.
INFO:tensorflow:Done with the session.
INFO:tensorflow:Loss for final step: 0.13476562.
INFO:tensorflow:training_loop marked as finished
Downloading file gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1057410.data-00000-of-00002 to /tmp/tmpf5yezk7i
Downloading file gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1057410.data-00001-of-00002 to /tmp/tmpf5yezk7i
Downloading file gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1057410.index to /tmp/tmpf5yezk7i
Downloading file gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/model.ckpt-1057410.meta to /tmp/tmpf5yezk7i
Building PyTorch model from configuration: T5Config {
  "architectures": [
    "T5WithLMHeadModel"
  ],
  "d_ff": 3072,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_num_buckets": 32,
  "vocab_size": 32128
}

Save PyTorch model to /tmp/tmpf5yezk7i/__TEMP__CHECKPOINT__/pt-statedict-base_embeddings_only_standard_vocab_keep_all_ckpts-1057410.pth
Save PyTorch model to gs://ptt5-1/base_embeddings_only_standard_vocab_keep_all_ckpts/models/base/checkpoints_pytorch/pt-statedict-base_embeddings_only_standard_vocab_keep_all_ckpts-1057410.pth
marcospiau123@ptt5-vm5:~/PTT5/pretraining/bash$ git pull
remote: Enumerating objects: 15, done.[K
remote: Counting objects:   6% (1/15)[Kremote: Counting objects:  13% (2/15)[Kremote: Counting objects:  20% (3/15)[Kremote: Counting objects:  26% (4/15)[Kremote: Counting objects:  33% (5/15)[Kremote: Counting objects:  40% (6/15)[Kremote: Counting objects:  46% (7/15)[Kremote: Counting objects:  53% (8/15)[Kremote: Counting objects:  60% (9/15)[Kremote: Counting objects:  66% (10/15)[Kremote: Counting objects:  73% (11/15)[Kremote: Counting objects:  80% (12/15)[Kremote: Counting objects:  86% (13/15)[Kremote: Counting objects:  93% (14/15)[Kremote: Counting objects: 100% (15/15)[Kremote: Counting objects: 100% (15/15), done.[K
remote: Compressing objects:  25% (1/4)[Kremote: Compressing objects:  50% (2/4)[Kremote: Compressing objects:  75% (3/4)[Kremote: Compressing objects: 100% (4/4)[Kremote: Compressing objects: 100% (4/4), done.[K
Unpacking objects:  11% (1/9)   Unpacking objects:  22% (2/9)   Unpacking objects:  33% (3/9)   Unpacking objects:  44% (4/9)   Unpacking objects:  55% (5/9)   Unpacking objects:  66% (6/9)   Unpacking objects:  77% (7/9)   Unpacking objects:  88% (8/9)   remote: Total 9 (delta 5), reused 9 (delta 5), pack-reused 0[K
Unpacking objects: 100% (9/9)   Unpacking objects: 100% (9/9), done.
From github.com:dl4nlp-rg/PTT5
   30007a6..2a171e1  master     -> origin/master
Updating 30007a6..2a171e1
Fast-forward
 pretraining/argparse_dumps/base_embeddings_only_custom_vocab_keep_all_ckpts.json             |   14 [32m+[m
 pretraining/bash/train_base_embeddings_only_custom_vocab_keep_all_ckpts_run_1.sh             |    4 [32m+[m[31m-[m
 pretraining/logs_scripts/train_base_embeddings_only_custom_vocab_keep_all_ckpts_run_1.sh.log | 4947 [32m++++++++++++++++++++++++++++++++++++++++++++++++++[m
 3 files changed, 4964 insertions(+), 1 deletion(-)
 create mode 100644 pretraining/argparse_dumps/base_embeddings_only_custom_vocab_keep_all_ckpts.json
 mode change 100644 => 100755 pretraining/bash/train_base_embeddings_only_custom_vocab_keep_all_ckpts_run_1.sh
 create mode 100644 pretraining/logs_scripts/train_base_embeddings_only_custom_vocab_keep_all_ckpts_run_1.sh.log
marcospiau123@ptt5-vm5:~/PTT5/pretraining/bash$ git add[K[K[K[K[K[K[Kexit
exit

Script done on 2020-07-27 04:24:34+00:00 [COMMAND_EXIT_CODE="0"]
