


hyps = open('/home/yhli/164/codes/fairseq-gec/nlpcc18-char/transformer_vanilla/results_train/hyp.word.tok').readlines()
refs = open('/home/yhli/164/datas/nlpcc18_121w/train_srcs_set/train.set.word.tok.tgt').readlines()

assert len(hyps) == len(refs)
