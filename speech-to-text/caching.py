import numpy as np
import librosa
import os
import scipy
import tqdm

sampling_rate = 22050
n_fft = 2048
frame_shift = 0.0125
frame_length = 0.05
hop_length = int(sampling_rate * frame_shift)
win_length = int(sampling_rate * frame_length)
n_mels = 80
reduction_factor = 5


def get_spectrogram(fpath):
    y, sr = librosa.load(fpath, sr = sampling_rate)
    D = librosa.stft(
        y = y, n_fft = n_fft, hop_length = hop_length, win_length = win_length
    )
    magnitude = np.abs(D)
    power = magnitude ** 2
    S = librosa.feature.melspectrogram(S = power, n_mels = n_mels)
    return np.transpose(S.astype(np.float32))


def reduce_frames(x, r_factor):
    T, C = x.shape
    num_paddings = reduction_factor - (T % r_factor) if T % r_factor != 0 else 0
    padded = np.pad(x, [[0, num_paddings], [0, 0]], 'constant')
    return np.reshape(padded, (-1, C * r_factor))


if not os.path.exists('spectrogram'):
    os.mkdir('spectrogram')

tolong_sebut = [
    'tolong-sebut/' + i for i in os.listdir('tolong-sebut') if '.wav' in i
]
sebut_perkataan_man = [
    'sebut-perkataan-man/' + i
    for i in os.listdir('sebut-perkataan-man')
    if '.wav' in i
]
sebut_perkataan_woman = [
    'sebut-perkataan-woman/' + i
    for i in os.listdir('sebut-perkataan-woman')
    if '.wav' in i
]

wavs = tolong_sebut + sebut_perkataan_man + sebut_perkataan_woman

for path in tqdm.tqdm(wavs):
    try:
        root, ext = os.path.splitext(path)
        root = root.replace('/', '-')
        spectrogram = get_spectrogram(path)
        spectrogram = reduce_frames(spectrogram, reduction_factor)
        np.save('spectrogram/%s.npy' % (root), spectrogram)
    except Exception as e:
        print(e)
        pass
