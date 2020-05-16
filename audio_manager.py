import numpy as np
import librosa
from scipy.signal import butter, sosfilt, correlate
from scipy.io import wavfile


def butter_bandpass(lowcut, highcut, sample_frequency, order=5):
    nyq = 0.5 * sample_frequency
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], analog=False, btype='band', output='sos')


def process(signal, bands, sample_frequency=44100, order=3, band_width=50):
    signals = []
    for i in range(0, len(bands)):
        sos = butter_bandpass(bands[i] - band_width, bands[i] + band_width, sample_frequency, order)
        signals.append(sosfilt(sos, signal))

    cor = signals[0]

    for i in range(1, len(signals)):
        cor += signals[i]

    return cor


def save_signal(output_file_path, signal, samplerate=44100):
    conv_signal = signal.astype(np.float32)
    wavfile.write(output_file_path, samplerate, conv_signal)


def get_from_file(input_file_path):
    return librosa.load(input_file_path, sr=None)


def get_from_mic(buffer_size=256, samplerate=44100):
    return None


def retrieve_audio(input_file_path=''):
    if input_file_path == '':
        return get_from_mic()
    else:
        return get_from_file(input_file_path)


def chop_in_frames(signal, frame_number):
    return np.split(signal, frame_number)


def extract_features(sample_rate, signal):
    return librosa.feature.mfcc(signal, sample_rate)
