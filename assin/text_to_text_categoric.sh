# Small
python3 t5_assin.py assin2_t5_small_entail_gen --model_name t5-small --vocab_name t5-small --bs 32 --architecture categoric_gen --max_epochs 50 --nout 2
python3 t5_assin.py assin2_t5_small_entail_acc --model_name t5-small --vocab_name t5-small --bs 32 --architecture categoric --max_epochs 50 --nout 2

# Base
python3 t5_assin.py assin2_t5_base_entail_gen --model_name t5-base --vocab_name t5-base --bs 2 --architecture categoric_gen --max_epochs 50 --nout 2
python3 t5_assin.py assin2_t5_base_entail_acc --model_name t5-base --vocab_name t5-base --bs 2 --architecture categoric --max_epochs 50 --nout 2
