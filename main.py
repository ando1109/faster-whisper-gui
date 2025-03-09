import tkinter as tk
import threading
import numpy as np
import sounddevice as sd
import librosa
import time
from faster_whisper import WhisperModel

# Initialize global variables
listening = False
audio_stream = None
recorded_audio = []
last_audio_time = time.time()
silence_threshold = 0.01  # Threshold for detecting silence
silence_duration = 1.5    # Duration of silence (in seconds)

# Model initialization
model_path = "large-v3-turbo"
model = WhisperModel(model_path, device="cuda", compute_type="float32")

# Function to check audio level
def check_silence(audio_chunk):
    global last_audio_time
    volume_norm = np.linalg.norm(audio_chunk)
    if volume_norm > silence_threshold:
        last_audio_time = time.time()

# sounddevice callback function
def audio_callback(indata, frames, time_info, status):
    global recorded_audio
    if status:
        print("Audio Status:", status)
    recorded_audio.append(indata.copy())
    check_silence(indata)

# Audio processing
def process_recording(text_widget, audio_data):
    process_data = audio_data.flatten()
    process_data = librosa.resample(process_data, orig_sr=48000, target_sr=16000)

    segments, _ = model.transcribe(
        process_data,
        beam_size=10,
        best_of=10,
        temperature=0.0,
        language="ja"
    )

    for segment in segments:
        text_widget.insert(tk.END, segment.text.strip() + "\n")

# Monitoring thread (silence detection)
def monitor_audio(text_widget):
    global listening, recorded_audio, audio_stream
    while listening:
        if time.time() - last_audio_time > silence_duration and recorded_audio:
            audio_stream.stop()
            audio_stream.close()
            text_widget.insert(tk.END, "Processing audio...\n")
            audio_data = np.concatenate(recorded_audio, axis=0)
            recorded_audio = []
            threading.Thread(target=process_recording, args=(text_widget, audio_data), daemon=True).start()
            
            # Restart recording after processing
            start_listening(text_widget)
        time.sleep(0.1)

# Start recording
def start_listening(text_widget):
    global listening, audio_stream, recorded_audio, last_audio_time
    if listening:
        return

    listening = True
    recorded_audio = []
    last_audio_time = time.time()

    device_index = 24  # Adjust this to your environment
    wasapi_settings = sd.WasapiSettings(auto_convert=True)

    audio_stream = sd.InputStream(
        device=device_index,
        samplerate=48000,
        channels=2,
        callback=audio_callback,
        extra_settings=wasapi_settings
    )
    audio_stream.start()

    threading.Thread(target=monitor_audio, args=(text_widget,), daemon=True).start()
    text_widget.insert(tk.END, "Listening started...\n")

# Stop recording
def stop_listening(text_widget):
    global listening, audio_stream
    if not listening:
        return
    listening = False
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
    text_widget.insert(tk.END, "Listening stopped.\n")

# Tkinter GUI setup
root = tk.Tk()
root.title("PC Audio Real-Time Transcription")

text_box = tk.Text(root, wrap="word", height=20, width=60)
text_box.pack(padx=10, pady=10)

start_button = tk.Button(root, text="Start Listening", command=lambda: start_listening(text_box))
start_button.pack(side="left", padx=10, pady=10)

stop_button = tk.Button(root, text="Stop Listening", command=lambda: stop_listening(text_box))
stop_button.pack(side="left", padx=10, pady=10)

root.mainloop()
