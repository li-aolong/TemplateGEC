checkpoint_path="scripts/model_t5/baseline"
output_dir=${checkpoint_path}/results

# bea-test: scripts/datas/t5_datas/bea-test.detok.json
# conll14-test: scripts/datas/t5_datas/conll14-test.detok.json
test_file="scripts/datas/t5_datas/bea-test.detok.json"

mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=0 python transformers/examples/pytorch/translation/run_maxtokens_translation.py \
    --model_name_or_path $checkpoint_path \
    --do_predict \
    --source_lang src \
    --target_lang tgt \
    --per_device_eval_batch_size 8 \
    --max_tokens_per_batch 1024 \
    --source_prefix "translate English to English: " \
    --test_file $test_file \
    --output_dir $output_dir \
    --num_beams=5 \
    --overwrite_output_dir \
    --predict_with_generate \
    
