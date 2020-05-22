import numpy as np
import audio_manager as audio
import matplotlib.pyplot as plt
import librosa.display
import frequency_band as band

file_name = "harvard_sentence_0"
sample_rate = 44100

# Get signal from file (or no parameter for mic)
signal, sample_rate = audio.retrieve_audio('TestSamples/' + file_name + '.wav')
# signal = audio.retrieve_audio()

# Apply Filtering
b1 = band.FrequencyBand(215, 50, 3)
b2 = band.FrequencyBand(410, 100, 3)
b3 = band.FrequencyBand(1100, 100, 1)
b4 = band.FrequencyBand(3400, 100, 1)

processed_signal = audio.process(signal, [b1, b2, b3, b4], sample_rate)

# Save processed Signal (OPTIONAL)
audio.save_signal('TestSamples/' + file_name + '_OUT.wav', processed_signal, sample_rate)

# # Chop in frames
# frames, added_samples = audio.chop_in_frames(processed_signal, sample_rate / 4)
#
# # Extract Features
# features = []
# for i in range(0, len(frames)):
#     features.append(audio.extract_features(sample_rate, frames[i]))
#
# # Print Results
# print("Sample Rate     : ", sample_rate, "Hz")
# print("Raw Signal Size : ", len(signal), "samples")
# print("Raw Signal Time : ", len(signal) / sample_rate, "seconds")
# print("Added", added_samples, "samples to last frame (total signal length :", added_samples + len(signal), "samples)")
# print("NB Frames       : ", len(frames))
# print("Frame size      : ", len(frames[0]), "samples")
# print("Frame time      : ", len(frames[0]) / sample_rate, "seconds")
#
# librosa.display.specshow(features[0], sr=sample_rate, x_axis='time')
# plt.show()
