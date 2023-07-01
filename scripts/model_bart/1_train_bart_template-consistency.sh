CODE_DIR="fairseq-0.12.2"

pretrain_model=/PATH/TO/bart_large

task="translation_consistance"
arch="bart_large"


lr=1e-5
max_tokens=1800
max_epoch=20
warmup_updates=5000

log_interval=200
update_freq=1

data_bin="scripts/model_bart/data-bins/template-consistency"
exp_path=bart_template-consistency

CKPTS=${exp_path}/ckpts
LOGS=${exp_path}/logs
mkdir -p $CKPTS
mkdir -p $LOGS


CUDA_VISIBLE_DEVICES=0 python $CODE_DIR/fairseq_cli/train.py $data_bin \
        --restore-file $pretrain_model \
        --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
        --arch ${arch} --share-all-embeddings \
        --task ${task}  \
        --source-lang src --target-lang tgt \
        --seed 32 --fp16 \
        --lr-scheduler inverse_sqrt \
        --lr $lr \
        --warmup-updates $warmup_updates \
        --weight-decay 0.00 --clip-norm 0.0 --dropout 0.3 \
        --attention-dropout 0.1 \
        --update-freq $update_freq  \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --save-dir $CKPTS \
        --tensorboard-logdir $LOGS \
        --criterion label_smoothed_cross_entropy_consistance --label-smoothing 0.1 \
        --no-progress-bar --log-format simple \
        --log-interval $log_interval \
        --save-interval 1 \
        --ddp-backend no_c10d \
        --max-epoch $max_epoch \
        --layernorm-embedding \
        --share-decoder-input-output-embed \
        --log-file $LOGS/train.log.file \
        --alpha 1 \
        --lossname consistance \
        --component decoder \
        --max-tokens $max_tokens \
        | tee -a $LOGS/train.log 2>&1