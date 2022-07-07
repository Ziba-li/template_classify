python run.py \
  --model_name_or_path model/bert_base_fintune1\
  --train_file data/emotion/cleaned/train1.csv \
  --validation_file data/emotion/cleaned/dev1.csv \
  --test_file data/emotion/cleaned/test1.csv \
  --do_train \
  --do_eval \
  --do_predict \
  --pad_to_max_length \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --save_total_limit 5 \
  --dataloader_num_workers 1 \
  --output_dir result1 \
#  --fp16
