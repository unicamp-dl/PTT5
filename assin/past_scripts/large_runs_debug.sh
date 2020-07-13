# T5 Large Categoric/Similarity experiments
python3 t5_assin.py assin2_t5_large_entail --model_name t5-large --vocab_name t5-large --bs 1 --architecture categoric --max_epochs 50 --nout 2 --debug
python3 t5_assin.py assin2_t5_large_long --model_name t5-large --vocab_name t5-large --bs 1 --architecture long --max_epochs 50 --debug

# PTT5 Similarity Large experiments
python3 t5_assin.py assin2_ptt5_large_long_custom_vocab --model_name ptt5-custom-vocab-large --vocab_name custom --bs 1 --architecture long --max_epochs 50 --debug
python3 t5_assin.py assin2_ptt5_large_long --model_name ptt5-standard-vocab-large --vocab_name t5-large --bs 1 --architecture long --max_epochs 50 --debug

# PTT5 Categoric Large experiments
python3 t5_assin.py assin2_ptt5_large_entail_custom_vocab --model_name ptt5-custom-vocab-large --vocab_name custom --bs 1 --architecture categoric --max_epochs 50 --nout 2 --debug
python3 t5_assin.py assin2_ptt5_large_entail --model_name ptt5-standard-vocab-large --vocab_name t5-large --bs 1 --architecture categoric --max_epochs 50 --nout 2 --debug
