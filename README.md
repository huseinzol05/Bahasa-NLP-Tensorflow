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
  * [Dependency Parsing](https://github.com/huseinzol05/Bahasa-Models-Tensorflow#dependency-parsing)
  * [Entity Tagging](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#entity-tagging)
  * [Abstractive Summarization](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#abstractive-summarization)
  * [Extractive Summarization](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#extractive-summarization)
  * [POS Tagging](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#pos-tagging)
  * [Optical Character Recognition](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#optical-character-recognition)
  * [Question-Answer](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#question-answer)
  * [Speech to Text](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#speech-to-text)
  * [Stemming](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#stemming)
  * [Topic Generator](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#topic-generator)
  * [Text to Speech](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#text-to-speech)
  * [Topic Modeling](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#topic-modeling)
  * [Word Vector](https://github.com/huseinzol05/Bahasa-NLP-Tensorflow#word-vector)

### [Augmentation](augmentation)

1. word2vec Malaya

### [Sparse classification](sparse-classification)

1. Fast-text Ngrams

### [Normal-text classification](normal-text-classification)

1. RNN LSTM + Bahdanau Attention, test accuracy 84%
2. RNN LSTM + Luong Attention, test accuracy 82%
3. Transfer-learning Multilanguage BERT, test accuracy 90.51%

70+ more models can get from [here](https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/text-classification).

### [Long-text classification](long-text-classification)

1. Dilated CNN, test accuracy 74%
2. Wavenet, test accuracy 68%

### [Dependency Parsing](dependency-parsing)

1. Bidirectional LSTM + CRF
2. Bidirectional LSTM + CRF + Bahdanau
3. Bidirectional LSTM + CRF + Luong

### [Entity Tagging](entity-tagging)

1. Bidirectional LSTM + CRF
2. Bidirectional LSTM + CRF + Bahdanau
3. Bidirectional LSTM + CRF + Luong

### [POS Tagging](pos-tagging)

1. Bidirectional LSTM + CRF
2. Bidirectional LSTM + CRF + Bahdanau
3. Bidirectional LSTM + CRF + Luong

### [Abstractive Summarization](abstractive-summarization)

1. Dilated Seq2Seq, train accuracy 83.13%
2. Pointer Generator + Bahdanau Attention, train accuracy 41.69%
3. Pointer Generator + Luong Attention, train accuracy 69%
4. Dilated Seq2Seq + Self Attention, train accuracy 58.07%
5. Dilated Seq2Seq + Self Attention + Pointer Generator, train accuracy 73.64%

### [Extractive Summarization](extractive-summarization)

1. Skip-thought
2. Residual Network + Bahdanau Attention

### [Optical Character Recognition](optical-character-recognition)

1. CNN + LSTM RNN

### [Question-Answer](question-answer)

1. End-to-End + GRU
2. Dynamic Memory + GRU

### [Speech to Text](speech-to-text)

1. BiRNN + LSTM + CTC Greedy
2. Wavenet
3. Deep speech 2

### [Text to Speech](text-to-speech)

1. Tacotron
2. Seq2Seq + Bahdanau Attention
3. Deep CNN + Monothonic Attention + Dilated CNN vocoder

### [Stemming](stemming)

1. Seq2seq + Beam decoder
2. Seq2seq + Bahdanau Attention + Beam decoder
3. Seq2seq + Luong Attention + Beam decoder

### [Topic Generator](topic-generator)

1. TAT-LSTM
2. TAV-LSTM
3. MTA-LSTM

### [Topic Modeling](topic-modeling)

1. Lda2Vec

### [Word Vector](word-vector)

1. word2vec
2. ELMO
3. Fast-text
