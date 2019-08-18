## How-to

1. Install required libraries,
```bash
pip3 install librosa numpy scipy
```

2. Download dataset from here, https://s3-ap-southeast-1.amazonaws.com/malaya-dataset/speech-bahasa.zip,
also https://raw.githubusercontent.com/huseinzol05/Malaya-Dataset/master/speech/train-test.json

3. Unzip it, and you will get 3 folders,
```text
sebut-perkataan-man
sebut-perkataan-woman
tolong-sebut
```

4. Optional, run [augmentation.py](augmentation.py) to increase the dataset size,
```bash
python3 augmentation.py
```

```text
random_strech =  0.5813149151887684
resample length_change =  0.8378528266260151
timeshift_fac =  0.19795814955092916
```

You can check what happened inside [augmentation.py](augmentation.py) from [augmentation.ipynb](augmentation.ipynb).

5. Run [caching.py](caching.py) to cache spectrogram locally,
```bash
python3 caching.py
```

```text
1%|▉                          | 113/17399 [00:24<1:00:05,  4.79it/s]
```

6. Run any notebook using Jupyter Notebook.

**For more models, you can check in https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/speech-to-text, but the dataset is not Bahasa Malaysia**
