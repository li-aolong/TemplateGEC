CODE_DIR="fairseq-0.12.2"


dest_data_bin="scripts/model_bart/data-bins/template-only"

python $CODE_DIR/fairseq_cli/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref scripts/datas/bart_datas/train.2-class.bart-bpe \
    --validpref scripts/datas/bart_datas/valid.2-class.bart-bpe \
    --destdir $dest_data_bin \
    --workers 30 \
    --srcdict scripts/datas/bart_datas/dict.txt \
    --joined-dictionary
