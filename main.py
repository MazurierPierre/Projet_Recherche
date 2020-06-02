import audio_manager as audio
import matplotlib.pyplot as plt
import librosa.display
import frequency_band as band
import neural_networks as NN


file_name = "test_samples/vsauce/vsauce_#1.wav"
sample_rate = 8000

# Get signal from file (or no parameter for mic)
signal, sample_rate = audio.retrieve_audio(file_name, sample_rate)
# signal = audio.retrieve_audio('', sample_rate)

# Apply Filtering
b1 = band.FrequencyBand(1000, 999, 1)
processed_signal = audio.process(signal, [b1], sample_rate)

# Save processed Signal (Optional)
processed_file_name = file_name + "_filtered.wav"
audio.save_signal(processed_file_name, processed_signal, sample_rate)

# Chop in frames
frames, added_samples = audio.chop_in_frames(processed_signal, sample_rate / 4)

# Extract Features
features = []
for i in range(0, len(frames)):
    features.append(audio.extract_features(sample_rate, frames[i]))

# Predict with neural network
# Since a high level API was used, it will take a file name as an input instead of array of features
result, confidence = NN.recognize(processed_file_name, verbose=True)

# Print Results (Optional)
print("Sample Rate     : ", sample_rate, "Hz")
print("Raw Signal Size : ", len(signal), "samples")
print("Raw Signal Time : ", len(signal) / sample_rate, "seconds")
print("Added", added_samples, "samples to last frame (total signal length :", added_samples + len(signal), "samples)")
print("NB Frames       : ", len(frames))
print("Frame size      : ", len(frames[0]), "samples")
print("Frame time      : ", len(frames[0]) / sample_rate, "seconds")

librosa.display.specshow(features[0], sr=sample_rate, x_axis='time')
plt.show()
