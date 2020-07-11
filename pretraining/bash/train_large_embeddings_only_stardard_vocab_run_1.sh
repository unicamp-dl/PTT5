python3 ../python/train_v2.py \
-b 1 \
-n large_embeddings_only_standard_vocab \
-ms large \
-e 1 \
-s 512 \
-jc '../../assin/T5_configs_json/ptt5-standard-vocab-large-config.json' \
--train_embedding_only
