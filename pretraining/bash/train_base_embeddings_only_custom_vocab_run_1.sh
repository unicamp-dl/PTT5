python3 ../python/train_v2.py \
-b 1 \
-n base_embeddings_only_custom_sentencepiece_vocab \
-ms base \
-e 4 \
-s 512 \
-jc '../../assin/T5_configs_json/ptt5-standard-vocab-base-config.json' \
--train_embedding_only \
-spm 'gs://ptt5-1/vocabs/spm_32000_unigram/spm_32000_pt.model'
