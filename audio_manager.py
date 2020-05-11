import numpy as np
import librosa


def process(signal):
    return signal


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
