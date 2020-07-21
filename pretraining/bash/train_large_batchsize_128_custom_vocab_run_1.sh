python3 ../python/train_v2.py \
-b 0.5 \
-n large_batchsize_128_custom_sentencepiece_vocab \
-ms large \
-e 2 \
-s 512 \
-jc '../../assin/T5_configs_json/ptt5-standard-vocab-large-config.json' \
-spm 'gs://ptt5-1/vocabs/spm_32000_unigram/spm_32000_pt.model' 
