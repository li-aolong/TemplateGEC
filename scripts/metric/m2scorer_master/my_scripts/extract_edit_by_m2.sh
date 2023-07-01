
stage="valid"

# python2 ../scripts/edit_creator.py \
#     --output /home/yhli/164/codes/VecConstNMT/My_scripts/nlpcc18_121w-char/transformer_vanilla_2/results_20/hyp.word.tok.m2 \
#     /home/yhli/164/codes/VecConstNMT/My_scripts/nlpcc18_121w-char/transformer_vanilla_2/results_20/hyp.word.tok \
#     /home/yhli/164/datas/nlpcc18_121w/test/test.word.tok.tgt


python2 /home/yhli/codes/m2scorer_master/scripts/edit_creator.py \
    /home/yhli/datas/nlpcc18_121w/test/test.word.tok.src \
    /home/yhli/datas/nlpcc18_121w/test/test.word.tok.tgt \
    --output ./tmp.m2 \
    # /home/yhli/164/datas/nlpcc18_121w/train_srcs_set/train.set.word.tok.tgt