import numpy as np
import audio_manager as audio
import matplotlib.pyplot as plt
import librosa.display

# Get signal from file (or no parameter for mic)
signal, sample_rate = audio.retrieve_audio('in.wav')

# Apply Filtering
processed_signal = audio.process(signal, [200, 500, 1000, 5000], sample_rate)

# Save processed Signal (OPTIONAL)
audio.save_signal('out.wav', processed_signal, sample_rate)

# # Chop in frames
# frames = audio.chop_in_frames(processed_signal, 3)
#
# # Extract Features
# features = []
# for i in range(0, len(frames)):
#     features.append(audio.extract_features(sample_rate, frames[i]))
#
# # Print Results
# print("Sample Rate : ", sample_rate, "Hz")
# print("Signal Size : ", len(signal), "samples")
# print("Signal Time : ", len(signal) / sample_rate, "seconds")
# print("NB Frames   : ", len(frames))
# print("Frame size  : ", len(frames[0]), "samples")
# print("Frame time  : ", len(frames[0]) / sample_rate, "seconds")
#
# librosa.display.specshow(features[0], sr=sample_rate, x_axis='time')
# plt.show()
