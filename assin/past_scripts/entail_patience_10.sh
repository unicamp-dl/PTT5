# Base
python3 t5_assin.py assin2_t5_base_entail --model_name t5-base --vocab_name t5-base --bs 2 --architecture categoric --max_epochs 50 --nout 2 --patience 10
python3 t5_assin.py assin2_ptt5_base_entail --model_name ptt5-4epoch-standard-vocab-base --vocab_name t5-base --bs 2 --architecture categoric --max_epochs 50 --nout 2 --patience 10
python3 t5_assin.py assin2_ptt5_base_entail_custom --model_name ptt5-custom-vocab-base --vocab_name custom --bs 2 --architecture categoric --max_epochs 50 --nout 2 --patience 10

# Small
python3 t5_assin.py assin2_t5_small_entail --model_name t5-small --vocab_name t5-small --bs 32 --architecture categoric --max_epochs 50 --nout 2 --patience 10
python3 t5_assin.py assin2_ptt5_small_entail --model_name ptt5-4epoch-standard-vocab-small --vocab_name t5-small --bs 32 --architecture categoric --max_epochs 50 --nout 2 --patience 10
python3 t5_assin.py assin2_ptt5_small_entail_custom --model_name ptt5-custom-vocab-small --vocab_name custom --bs 32 --architecture categoric --max_epochs 50 --nout 2 --patience 10
