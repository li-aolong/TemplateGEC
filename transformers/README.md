# Exploiting Translationese for GEC on (m)T5-Large Pre-trained Language Models 
If you want to use the obtained translationese as input for data augmentation for GEC on `(m)T5 large` pre-trained language models, please follow the instructions below.

## Requirements and Installation

This implementation is based on huggingface/[transformers(v4.13.0)](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/) version >= 1.3.1
- Python version >= 3.6

```
cd transformer
pip install .
pip install -r requirements.txt
```
## Prepare the detokenizer for GEC languages
Download [Moses Detokenizer](https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/detokenizer.perl) for English/German/Russian.
```
sh  detokenization.sh
```

## Data Preprocessing
The train/dev/test sets need to be converted to custom JSONLINES files. Our synthetic data with a `special tag` at the beginning of every source sentence.
```
sh tsv2json.sh
```
## Training
```
sh /shell_ftT5/train_en.sh
sh /shell_ftT5/train_de.sh
sh /shell_ftT5/train_ru.sh
sh /shell_ftT5/train_zh.sh
```

## Generation and Evaluation
```
sh /shell_ftT5/Generate_evaluate_en.sh
sh /shell_ftT5/Generate_evaluate_de.sh
sh /shell_ftT5/Generate_evaluate_ru.sh
sh /shell_ftT5/Generate_evaluate_zh.sh
```
Note: The tokenizier for BEA2019 English using [spaCy(v1.9.0)](https://spacy.io/).

  
  
