<p align="center">
    <a href="#readme">
        <img alt="logo" width="30%" src="cintailah-bahasa-malaysia-menggunakan-tensorflow.jpg">
    </a>
</p>
<p align="center">
  <a href="https://github.com/huseinzol05/Bahasa-NLP-Tensorflow/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

---

**Bahasa-NLP-Tensorflow**, Gathers Tensorflow deep learning models for Bahasa Malaysia NLP problems, **code simplify inside Jupyter Notebooks 100% including dataset**.

## Table of contents
  * [Augmentation](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#augmentation)
  * [Sparse classification](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#sparse-classification)
  * [Long-text classification](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#long-text-classification)
  * [Dependency Parsing](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#dependency-parsing)
  * [English-Malay Translation](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#english-malay-translation)
  * [Entity Tagging](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#entity-tagging)
  * [Abstractive Summarization](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#abstractive-summarization)
  * [Extractive Summarization](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#extractive-summarization)
  * [POS Tagging](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#pos-tagging)
  * [Optical Character Recognition](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#optical-character-recognition)
  * [Question-Answer](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#question-answer)
  * [Semantic Similarity](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#semantic-similarity)
  * [Speech to Text](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#speech-to-text)
  * [Stemming](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#stemming)
  * [Topic Generator](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#topic-generator)
  * [Text to Speech](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#text-to-speech)
  * [Topic Modeling](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#topic-modeling)
  * [Word Vector](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#word-vector)

### [Augmentation](augmentation)

1. word2vec Malaya

### [Sparse classification](sparse-classification)

Trained on [Tatoeba dataset](http://downloads.tatoeba.org/exports/sentences.tar.bz2).

1. Fast-text Ngrams, test accuracy 88%

### [Normal-text classification](normal-text-classification)

Trained on [Bahasa subjectivity dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/subjectivity).

1. RNN LSTM + Bahdanau Attention, test accuracy 84%
2. RNN LSTM + Luong Attention, test accuracy 82%
3. Transfer-learning Multilanguage BERT, test accuracy 94.88%

70+ more models can get from [here](https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/text-classification).

### [Long-text classification](long-text-classification)

Trained on [Bahasa fakenews dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/fake-news).

1. Dilated CNN, test accuracy 74%
2. Wavenet, test accuracy 68%
3. BERT Multilanguage, test accuracy 85%
4. BERT-Bahasa Base, test accuracy 88%

### [Dependency Parsing](dependency-parsing)

Trained on [Bahasa dependency parsing dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/dependency).

1. Bidirectional LSTM + CRF
2. Bidirectional LSTM + CRF + Bahdanau
3. Bidirectional LSTM + CRF + Luong

### [English-Malay Translation](english-malay-translation)

Trained on [100k english-malay dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/english-malay).

1. Attention is All you need, train accuracy 19.09% test accuracy 20.38%
2. BiRNN Seq2Seq Luong Attention, Beam decoder, train accuracy 45.2% test accuracy 37.26%
3. Convolution Encoder Decoder, train accuracy 35.89% test accuracy 30.65%
4. Dilated Convolution Encoder Decoder, train accuracy 82.3% test accuracy 56.72%
5. Dilated Convolution Encoder Decoder Self-Attention, train accuracy 60.76% test accuracy 36.59%

### [Entity Tagging](entity-tagging)

Trained on [Bahasa entity dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/entities).

1. Bidirectional LSTM + CRF, test accuracy 95.10%
2. Bidirectional LSTM + CRF + Bahdanau, test accuracy 94.34%
3. Bidirectional LSTM + CRF + Luong, test accuracy 94.84%
4. BERT Multilanguage, test accuracy 96.43%
5. BERT-Bahasa Base, test accuracy 98.11%
6. BERT-Bahasa Small, test accuracy 98.47%
7. XLNET-Bahasa Base, test accuracy 31.47%

### [POS Tagging](pos-tagging)

Trained on [Bahasa entity dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/part-of-speech).

1. Bidirectional LSTM + CRF
2. Bidirectional LSTM + CRF + Bahdanau
3. Bidirectional LSTM + CRF + Luong

### [Abstractive Summarization](abstractive-summarization)

Trained on [Malaysia news dataset](https://github.com/huseinzol05/Malaya-Dataset#30k-news).

Accuracy based on ROUGE-2 after 20 epochs only.

1. Dilated Seq2Seq, test accuracy 23.926%
2. Pointer Generator + Bahdanau Attention, test accuracy 15.839%
3. Pointer Generator + Luong Attention, test accuracy 26.23%
4. Dilated Seq2Seq + Pointer Generator, test accuracy 20.382%
5. BERT Multilanguage + Dilated Fairseq + Pointer Generator, test accuracy 23.7134%

### [Extractive Summarization](extractive-summarization)

Trained on [Malaysia news dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/news).

1. Skip-thought
2. Residual Network + Bahdanau Attention

### [Optical Character Recognition](optical-character-recognition)

1. CNN + LSTM RNN, test accuracy 91.22%

### [Question-Answer](question-answer)

Trained on [Bahasa QA dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/question-answer).

1. End-to-End + GRU, test accuracy 89.17%
2. Dynamic Memory + GRU, test accuracy 98.86%

### [Semantic Similarity](semantic-similarity)

Trained on [Translated Duplicated Quora question dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/text-similarity/quora).

1. LSTM Bahdanau + Contrastive loss, test accuracy 79%
2. Dilated CNN + Contrastive loss, test accuracy 77%
3. Self-Attention + Contrastive loss, test accuracy 77%
4. BERT + Cross entropy, test accuracy 83%

### [Speech to Text](speech-to-text)

Trained on [Kamus speech dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/speech).

1. BiRNN + LSTM + CTC Greedy, test accuracy 72.03%
2. Wavenet, test accuracy 10.21%
3. Deep speech 2, test accuracy 56.51%
4. Dilated-CNN, test accuracy 59.31%

### [Text to Speech](text-to-speech)

1. Tacotron
2. Seq2Seq + Bahdanau Attention
3. Deep CNN + Monothonic Attention + Dilated CNN vocoder

### [Stemming](stemming)

Trained on [stemming dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/stemmer).

1. Seq2seq + Beam decoder
2. Seq2seq + Bahdanau Attention + Beam decoder
3. Seq2seq + Luong Attention + Beam decoder

### [Topic Generator](topic-generator)

Trained on [Malaysia news dataset](https://github.com/huseinzol05/Malaya-Dataset/tree/master/news).

1. TAT-LSTM, test accuracy 32.89%
2. TAV-LSTM, test accuracy 40.69%
3. MTA-LSTM, test accuracy 32.96%

### [Topic Modeling](topic-modeling)

1. Lda2Vec

### [Word Vector](word-vector)

1. word2vec
2. ELMO
3. Fast-text
