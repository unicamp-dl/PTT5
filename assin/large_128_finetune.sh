# Similarity
python3 t5_assin.py assin2_ptt5_large_1282_long_custom_vocab --model_name ptt5-custom-vocab-128-large --vocab_name custom --bs 1 --acum 2 --architecture long --max_epochs 50 --checkpoint_path checkpoints --model_path models --log_path logs
python3 t5_assin.py assin2_ptt5_large_128_long_custom_vocab --model_name ptt5-custom-vocab-128-large --vocab_name custom --bs 1 --architecture long --max_epochs 50 --checkpoint_path checkpoints --model_path models --log_path logs

# Classification
python3 t5_assin.py assin2_ptt5_large_1282_entail_custom_vocab --model_name ptt5-custom-vocab-128-large --vocab_name custom --bs 1 --acum 2 --architecture categoric --max_epochs 50 --nout 2 --checkpoint_path checkpoints --model_path models --log_path logs --patience 10
python3 t5_assin.py assin2_ptt5_large_128_entail_custom_vocab --model_name ptt5-custom-vocab-128-large --vocab_name custom --bs 1 --architecture categoric --max_epochs 50 --nout 2 --checkpoint_path checkpoints --model_path models --log_path logs --patience 10
