python3 ../python/train_v2.py \
-b 1 \
-ms base \
-e 1 \
-s 512 \
-jc '../../assin/T5_configs_json/ptt5-standard-vocab-base-config.json' \
--keep_all_checkpoints \
-n base_embeddings_only_custom_vocab_keep_all_ckpts  \
-spm 'gs://ptt5-1/vocabs/spm_32000_unigram/spm_32000_pt.model'  \
--train_embedding_only \
--save_checkpoints_steps 500

