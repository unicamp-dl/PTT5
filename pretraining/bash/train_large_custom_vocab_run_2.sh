python3 ../python/train_custom_vocab.py \
-b 1 \
-n large_custom_sentencepiece_vocab \
-ms large \
-e 2 \
-s 512 \
-jc '../../assin/T5_configs_json/ptt5-standard-vocab-large-config.json' \
-spm 'gs://ptt5-1/vocabs/spm_32000_unigram/spm_32000_pt.model' \
-bp gs://ptt5-1/large_custom_sentencepiece_vocab/models/
