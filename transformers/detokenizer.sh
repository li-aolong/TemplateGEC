

# for En/De/Ru
perl /Trans4GEC/script/mosesdetokenizer.perl -l en < tokenized_en-gec > detokenized_en-gec
perl /Trans4GEC/script/mosesdetokenizer.perl -l de < tokenized_de-gec > detokenized_de-gec
perl /Trans4GEC/script/mosesdetokenizer.perl -l ru < tokenized_ru-gec > detokenized_ru-gec

# for Zh

python /transformers/evaluation_scorer/tokenization/pkunlp/detok_zh.py char_tokenized_zh-gec  detokenized_zh-gec