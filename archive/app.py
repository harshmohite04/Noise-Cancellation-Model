import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile

# Read the wav file
sample_rate, audio = wavfile.read(r'D:\harsh\Code Playground\Innover8ers\Noise Cancellation\archive\noisy_trainset_wav\p226_004.wav')

# If the audio has more than one channel, use only the first channel
if audio.ndim > 1:
    audio = audio[:, 0]

# Generate time data for the x-axis
time = np.linspace(0, len(audio) / sample_rate, num=len(audio))

# Plot the waveform using seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))
sns.lineplot(x=time, y=audio, color='b')
plt.title("Waveform of the Audio File")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()