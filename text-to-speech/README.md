## How-to

1. Install required libraries,
```bash
pip3 install librosa numpy scipy
```

2. Download dataset from here, https://s3-ap-southeast-1.amazonaws.com/malaya-dataset/speech-bahasa.zip

3. Unzip it, and you will get 3 folders,
```text
sebut-perkataan-man
sebut-perkataan-woman
tolong-sebut
```

4. Run [caching.py](caching.py) to cache meg and mel locally,
```bash
python3 caching.py
```

```text
1%|▉                          | 113/17399 [00:24<1:00:05,  4.79it/s]
```

6. Run any notebook using Jupyter Notebook.

**For more models, you can check in https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/text-to-speech, but the dataset is not Bahasa Malaysia**
