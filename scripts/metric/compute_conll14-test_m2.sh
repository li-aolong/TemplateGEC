

hyp_file=/PATH/TO/HYP_FILE

gold="../datas/conll14/official-2014.combined.m2"

python ./m2scorer_master/scripts/m2scorer.py $hyp_file $gold \
    | tee ${hyp_file}.score