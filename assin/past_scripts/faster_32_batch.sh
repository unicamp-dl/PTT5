# Bigger batch faster
python3 t5_assin.py assin2_ptt5_base_acum_fast_long_custom_vocab --model_name ptt5-custom-vocab-base --vocab_name custom --bs 2 --acum 16 --architecture long --max_epochs 50 --lr 0.001

# Emb bigger batch faster
python3 t5_assin.py assin2_ptt5_base_emb_acum_fast_long_custom_vocab --model_name ptt5-custom-vocab-emb-base --vocab_name custom --bs 2 --acum 16  --architecture long --max_epochs 50 --lr 0.001