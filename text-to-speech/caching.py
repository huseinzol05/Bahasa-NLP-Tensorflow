import numpy as np
import librosa
import os
import scipy
import tqdm

sampling_rate = 22050
n_fft = 2048
frame_shift = 0.0125
frame_length = 0.05
fourier_window_size = 2048
max_db = 100
ref_db = 20
preemphasis = 0.97
hop_length = int(sampling_rate * frame_shift)
win_length = int(sampling_rate * frame_length)
n_mels = 80
resampled = 5
reduction_factor = 5


def get_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr = sampling_rate)
    y, _ = librosa.effects.trim(y)
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])
    linear = librosa.stft(
        y = y,
        n_fft = fourier_window_size,
        hop_length = hop_length,
        win_length = win_length,
    )
    mag = np.abs(linear)
    mel_basis = librosa.filters.mel(sampling_rate, fourier_window_size, n_mels)
    mel = np.dot(mel_basis, mag)
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)
    return mel.T.astype(np.float32), mag.T.astype(np.float32)


def load_file(path):
    mel, mag = get_spectrogram(path)
    t = mel.shape[0]
    num_paddings = resampled - (t % resampled) if t % resampled != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode = 'constant')
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode = 'constant')
    return mel.reshape((-1, n_mels * resampled)), mag


if not os.path.exists('mel'):
    os.mkdir('mel')
if not os.path.exists('mag'):
    os.mkdir('mag')

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
        mel, mag = load_file(path)
        root, ext = os.path.splitext(path)
        root = root.replace('/', '-')
        np.save('mel/%s.npy' % (root), mel)
        np.save('mag/%s.npy' % (root), mag)
    except Exception as e:
        print(e)
        pass
