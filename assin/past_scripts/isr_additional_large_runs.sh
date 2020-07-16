# 2pochs large
python3 t5_assin.py assin2_ptt5_large_entail_custom_vocab_10p_2poch --gpu 0 --model_name ptt5-2poch-custom-vocab-large --vocab_name custom --bs 1 --architecture categoric --max_epochs 50 --nout 2 --patience 10 --checkpoint_path ../checkpoints --model_path exp7 --log_path exp7

# patience 10 entails
python3 t5_assin.py assin2_t5_large_entail_10p --gpu 1 --model_name t5-large --vocab_name t5-large --bs 1 --architecture categoric --max_epochs 50 --nout 2 --patience 10 --checkpoint_path ../checkpoints --model_path exp8 --log_path exp8
python3 t5_assin.py assin2_ptt5_large_entail_custom_vocab_10p --gpu 2 --model_name ptt5-custom-vocab-large --vocab_name custom --bs 1 --architecture categoric --max_epochs 50 --nout 2 --patience 10 --checkpoint_path ../checkpoints --model_path exp9 --log_path exp9
python3 t5_assin.py assin2_ptt5_large_entail_10p --gpu 3 --model_name ptt5-standard-vocab-large --vocab_name t5-large --bs 1 --architecture categoric --max_epochs 50 --nout 2 --patience 10 --checkpoint_path ../checkpoints --model_path exp10 --log_path exp10
