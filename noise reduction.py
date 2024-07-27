import noisereduce as nr
import librosa
import soundfile as sf

# Load the audio file
audio, sr = librosa.load(r'C:\Users\Asus\OneDrive\Documents\python\Noise cancellation\Dataset\noisy_trainset_wav\p226_016.wav')

# Reduce the noise in the audio file
reduced_noise = nr.reduce_noise(audio, sr=sr)

# Save the noise-reduced audio file
sf.write('noise_reduced_audio_file.wav', reduced_noise, sr)


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(audio)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Original')
plt.show()



plt.figure(figsize = (15, 6))
plt.plot(reduced_noise)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Denoised')
plt.show()