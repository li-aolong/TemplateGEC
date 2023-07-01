CODE_DIR="./fairseq-0.12.2"

ckpt_name="checkpoint_best.pt"

# test or valid
gen_subset="test"

beam_size="1"
batch_size="256"

output_path="scripts/model_transformer/transformer_baseline"
result_file="result_bea-${gen_subset}"

data_bin="scripts/model_transformer/data-bins/transformer_clang8_data-bins/pred"

mkdir -p $output_path
mkdir -p $output_path/$result_file

# fairseq-generate
CUDA_VISIBLE_DEVICES=0 python $CODE_DIR/fairseq_cli/generate.py $data_bin \
    --path $output_path/ckpts/$ckpt_name \
    --results-path $output_path/$result_file \
    --batch-size $batch_size \
    --gen-subset $gen_subset \
    --beam $beam_size \
    --source-lang src --target-lang tgt \
    --remove-bpe
wait

grep ^H $output_path/$result_file/generate-${gen_subset}.txt | sort -n -k 2 -t '-' | cut -f 3 > $base_path/$results_path/${model_name}.${parameter}.output


