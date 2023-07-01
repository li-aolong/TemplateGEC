CODE_DIR="./fairseq-0.12.2"

arch="transformer_vaswani_wmt_en_de_big"

lr=5e-4
max_tokens=16384
max_epoch=30
warmup_updates=1600

data_bin="scripts/model_transformer/data-bins/transformer_clang8_data-bins/pred"

log_interval=100
update_freq=1

exp_path=transformer_baseline
CKPTS=${exp_path}/ckpts
LOGS=${exp_path}/logs
mkdir -p $CKPTS
mkdir -p $LOGS


CUDA_VISIBLE_DEVICES=0 python $CODE_DIR/fairseq_cli/train.py $data_bin \
        --fp16 --seed 32 \
        --share-all-embeddings \
        --lr-scheduler inverse_sqrt \
        --lr $lr \
        --warmup-init-lr 1e-07 --warmup-updates $warmup_updates \
        --weight-decay 0.0 --clip-norm 0.0 \
        --dropout 0.3 \
        --max-tokens $max_tokens --update-freq $update_freq \
        --arch $arch \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --save-dir $CKPTS \
        --tensorboard-logdir $LOGS \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --no-progress-bar --log-format simple \
        --log-interval $log_interval \
        --ddp-backend no_c10d \
        --max-epoch $max_epoch \
        --patience 3 \
        | tee -a $LOGS/train.log
