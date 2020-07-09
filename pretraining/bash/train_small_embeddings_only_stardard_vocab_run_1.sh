python3 ../python/train_v2.py \
-b 1 \
-n small_embeddings_only_standard_vocab \
-ms small \
-e 2 \
-s 512 \
-jc '../../assin/T5_configs_json/ptt5-standard-vocab-small-config.json' \
--train_embedding_only
