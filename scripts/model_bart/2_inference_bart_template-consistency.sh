path="scripts/model_bart/bart_template-consistency"

# bea-test: scripts/model_bart/datas/bea-test_2-class.src
# conll14-test: scripts/model_bart/datas/conll14-test_2-class.src
src_file="scripts/model_bart/datas/bea-test_2-class.src"

# bea-test: ${path}/results_bea-test
# conll14-test: ${path}/results_conll14-test
output_path="${path}/results_bea-test"

beam_size=1
batch_size=64

ckpt_file="checkpoint_best.pt"

mkdir -p $output_path
output_file=${output_path}/hyp.txt


CUDA_VISIBLE_DEVICES=0 python scripts/model_bart/inference_bart.py \
    --model-dir $path/ckpts \
    --model-file $ckpt_file \
    --src $src_file \
    --out $output_file \
    --beam-size $beam_size \
    --bsz ${batch_size}