python3 t5_assin.py --name assin2_small_gen --model_name t5-small --vocab_name t5-small --bs 32 --architecture gen --max_epochs 50
python3 t5_assin.py --name assin2_small_long --model_name t5-small --vocab_name t5-small --bs 32 --architecture long --max_epochs 50
python3 t5_assin.py --name assin2_base_gen --model_name t5-base --vocab_name t5-base --bs 2 --architecture gen --max_epochs 50
python3 t5_assin.py --name assin2_base_long --model_name t5-base --vocab_name t5-base --bs 2 --architecture long --max_epochs 50
