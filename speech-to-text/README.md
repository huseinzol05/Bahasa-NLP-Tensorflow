## How-to

1. Install required libraries,

```bash
pip3 install librosa numpy scipy
```

2. Run [generate-labels.ipynb](generate-labels.ipynb) to get [labels-text.txt](labels-text.txt)

2. Generate sentencepiece vocab,

```bash
spm_train \
--input=label-text.txt \
--model_prefix=sp10m.cased.speech \
--vocab_size=400 \
--character_coverage=0.99995 \
--model_type=unigram \
--control_symbols=\<cls\>,\<sep\>,\<pad\>,\<mask\>,\<eod\> \
--user_defined_symbols=\<eop\>,.,\(,\),\",-,–,£,€ \
--shuffle_input_sentence \
--input_sentence_size=10000000
```
