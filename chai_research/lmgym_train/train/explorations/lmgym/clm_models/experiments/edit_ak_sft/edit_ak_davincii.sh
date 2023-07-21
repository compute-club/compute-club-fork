deepspeed train.py \
  --model_name_or_path ChaiML/gptj_distilation_combine_loss_epoch2 \
  --tokenizer_name ChaiML/gptj_distilation_combine_loss_epoch2 \
  --dataset_name ChaiML/full_user_edit_responses-filtered-deduped-sampled-prepared \
  --train_to_probs False \
  --do_train \
  --do_eval \
  --logging_strategy steps \
  --evaluation_strategy steps \
  --eval_steps 2100 \
  --save_strategy epoch \
  --save_steps 1 \
  --logging_steps 250 \
  --logging_first_step \
  --report_to all \
  --output_dir /tmp/gptj_distilation_combine_loss_epoch2-14032023 \
  --overwrite_output_dir \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_eval_samples 500 \
  --num_train_epochs 4 \
  --eval_first_step False \
  --learning_rate 1e-6 \
  --fp16 \
  --seed 99 \
  --num_eval_prompts 128 \
  --validation_split_percentage 1 \
  --remove_unused_columns False \
  --deepspeed deepspeed_configs/ds_config_soft.json \
  --clean_enabled False \
  --add_reward_scores True \
  --block_size 512 \
  --eval_prompt_path examples.json
