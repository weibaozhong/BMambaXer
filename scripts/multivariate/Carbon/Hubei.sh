 export CUDA_VISIBLE_DEVICES=0

model_name=BMambaXer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/carbon/ \
  --data_path hubei.csv \
  --model_id hubei_96_7 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96\
  --pred_len 7 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 8\
  --itr 1 \
  --learning_rate 0.0001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/carbon/ \
  --data_path hubei.csv \
  --model_id hubei_96_14 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96\
  --pred_len 14 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 8\
  --itr 1 \
  --learning_rate 0.0001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/carbon/ \
  --data_path hubei.csv \
  --model_id hubei_96_30 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96\
  --pred_len 30 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 8\
  --itr 1 \
  --learning_rate 0.0001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/carbon/ \
  --data_path hubei.csv \
  --model_id hubei_96_45 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96\
  --pred_len 45 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 8\
  --itr 1 \
  --learning_rate 0.0001

