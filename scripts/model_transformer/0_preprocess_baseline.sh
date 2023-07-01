CODE_DIR="fairseq-0.12.2"


dest_data_bin="scripts/model_transformer/data-bins/transformer_clang8_data-bins/original"

python $CODE_DIR/fairseq_cli/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref scripts/datas/transformer_datas/train/train.tok.bpe \
    --validpref scripts/datas/transformer_datas/valid/valid.tok.bpe \
    --testpref scripts/datas/transformer_datas/test/test.tok.bpe \
    --destdir $dest_data_bin \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --workers 30 \
    --joined-dictionary \
