CODE_DIR="fairseq-0.12.2"

dest_data_bin="scripts/model_transformer/data-bins/transformer_clang8_data-bins/pred"
python $CODE_DIR/fairseq_cli/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref scripts/datas/transformer_datas/train/train.2-class.pred.tok.bpe \
    --validpref scripts/datas/transformer_datas/valid/valid.2-class.pred.tok.bpe \
    --testpref scripts/datas/transformer_datas/test/test.2-class.pred.tok.bpe \
    --destdir $dest_data_bin \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --workers 30 \
    --joined-dictionary \
