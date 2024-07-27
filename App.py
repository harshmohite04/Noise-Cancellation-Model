import pyaudio
import numpy as np
import tensorflow as tf
import threading
import queue

# Load the pre-trained model
autoencoder = tf.keras.models.load_model(r'C:\Users\Asus\OneDrive\Documents\python\Noise cancellation\denoising_autoencoder.h5')

# Parameters
CHUNK = 16384  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels (mono)
RATE = 44100  # Sampling rate (samples per second)

# Queue to hold audio chunks
audio_queue = queue.Queue()

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open input and output audio streams
stream_in = p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)

stream_out = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=CHUNK)

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

def process_audio():
    print("Real-time noise cancellation started...")
    try:
        while True:
            if not audio_queue.empty():
                data = audio_queue.get()
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Normalize and reshape for the model
                audio_data = audio_data / 32768.0
                audio_data = np.expand_dims(audio_data, axis=0)
                audio_data = np.expand_dims(audio_data, axis=-1)

                # Perform noise cancellation using the pre-trained model
                denoised_data = autoencoder.predict(audio_data)

                # Reshape and denormalize the output
                denoised_data = np.squeeze(denoised_data)
                denoised_data = (denoised_data * 32768.0).astype(np.int16)

                # Convert back to byte data
                denoised_data = denoised_data.tobytes()

                # Write to output audio stream
                stream_out.write(denoised_data)

    except KeyboardInterrupt:
        print("Stopping noise cancellation...")

    finally:
        # Close streams
        stream_in.stop_stream()
        stream_in.close()
        stream_out.stop_stream()
        stream_out.close()

        # Terminate PyAudio
        p.terminate()

# Set the input stream callback
stream_in.start_stream()
stream_out.start_stream()

# Start the audio processing thread
audio_thread = threading.Thread(target=process_audio)
audio_thread.start()

# Keep the main thread alive
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Stopping...")

# Stop the audio thread
audio_thread.join()
