python3 ../python/train_v2.py \
-b 1 \
-n small_embeddings_only_standard_vocab \
-bp 'gs://ptt5-1/small_embeddings_only_standard_vocab/models/' \
-ms small \
-e 2 \
-s 512 \
-jc '../../assin/T5_configs_json/ptt5-standard-vocab-small-config.json' \
--train_embedding_only
