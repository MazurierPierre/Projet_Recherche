import numpy as np
import librosa
from scipy.signal import butter, sosfilt
from scipy.io import wavfile
import sounddevice as sd


def butter_bandpass(lowcut, highcut, sample_frequency, order=5):
    nyq = 0.5 * sample_frequency
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], analog=False, btype='band', output='sos')


def process(signal, bands, sample_frequency=44100):
    signals = []

    for band in bands:
        print("Found band :", band.frequency, band.bandwidth, band.order)
        sos = butter_bandpass(
            band.frequency - band.bandwidth,
            band.frequency + band.bandwidth,
            sample_frequency,
            order=band.order
        )
        signals.append(sosfilt(sos, signal))

    processed_signal = signals[0]

    for i in range(1, len(signals)):
        processed_signal += signals[i]

    return processed_signal


def save_signal(output_file_path, signal, samplerate=44100):
    conv_signal = signal.astype(np.float32)
    wavfile.write(output_file_path, samplerate, conv_signal)


def get_from_file(input_file_path):
    return librosa.load(input_file_path, sr=None)


def get_from_mic(buffer_size=1024, samplerate=44100):
    try:
        sd.default.samplerate = samplerate
        sd.default.dtype = np.float32
        sd.default.channels = 1
        sd.default.blocksize = buffer_size

        recording = []

        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)

            for i in range(0, len(indata)):
                recording.extend(indata[i])

        with sd.Stream(callback=callback):
            input("Press any key to stop the recording ...")
            return np.asarray(recording)

    except Exception as e:
        print(e)


def retrieve_audio(input_file_path=''):
    if input_file_path == '':
        return get_from_mic()
    else:
        return get_from_file(input_file_path)


def chop_in_frames(signal, frame_size):

    added_samples = 0

    while len(signal) % frame_size != 0:
        signal = np.append(signal, 0)
        added_samples = added_samples + 1

    return np.split(signal, len(signal) / frame_size), added_samples


def extract_features(sample_rate, signal):
    return librosa.feature.mfcc(signal, sample_rate)
