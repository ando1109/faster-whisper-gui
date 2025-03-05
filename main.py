import tkinter as tk
import threading
import numpy as np
import sounddevice as sd
import librosa  # for resampling
from faster_whisper import WhisperModel

# Initialize global variables
listening = False         # Flag indicating whether recording is in progress
audio_stream = None       # Audio stream object
recorded_audio = []       # List to store the recorded audio data

# Initialize the model (adjust parameters as needed)
model_path = "large-v3-turbo"
model = WhisperModel(model_path, device="cuda", compute_type="float32")

# Callback function for sounddevice
def audio_callback(indata, frames, time, status):
    if status:
        print("Audio Status:", status)
    # If recording is active, append the audio data to the list
    if listening:
        recorded_audio.append(indata.copy())

# Function to process the complete audio data after recording stops
def process_recording(text_widget, audio_data):
    # Here, the two-dimensional data (samples x channels) is flattened for processing
    process_data = audio_data.flatten()
    INPUT_SR = 48000    # Capture sample rate
    TARGET_SR = 16000   # Sample rate expected by the model
    # Resample from 48000 Hz to 16000 Hz
    process_data = librosa.resample(process_data, orig_sr=INPUT_SR, target_sr=TARGET_SR)
    # Specify language="ja" for Japanese recognition (also adjust beam_size, best_of, etc.)
    segments, info = model.transcribe(
        process_data,
        beam_size=10,
        best_of=10,
        temperature=0.0,
        language="ja"
    )
    # Display each transcribed segment in the text widget
    for segment in segments:
        text = segment.text.strip()
        text_widget.insert(tk.END, text + "\n")

# Handler for when the Start Recording button is pressed
def start_listening(text_widget):
    global listening, audio_stream, recorded_audio
    if listening:
        return
    listening = True
    recorded_audio = []  # Clear any previous recording data
    text_widget.insert(tk.END, "Starting recording...\n")
    # For VB‑CABLE, specify "CABLE Output (VB-Audio Virtual Cable)" as the capture device
    device_index = 24  # Change this index according to the one found with sd.query_devices()
    # WASAPI settings (auto_convert=True)
    wasapi_settings = sd.WasapiSettings(auto_convert=True)
    # Set the number of channels to match the VB‑CABLE output (usually 2)
    audio_stream = sd.InputStream(
        device=device_index,
        samplerate=48000,
        channels=2,
        callback=audio_callback,
        extra_settings=wasapi_settings
    )
    audio_stream.start()

# Handler for when the Stop Recording button is pressed
def stop_listening(text_widget):
    global listening, audio_stream, recorded_audio
    if not listening:
        return
    listening = False
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
    text_widget.insert(tk.END, "Recording stopped. Processing audio...\n")
    # If recorded data exists, concatenate and process it in a separate thread to avoid blocking the UI
    if recorded_audio:
        audio_data = np.concatenate(recorded_audio, axis=0)
        threading.Thread(target=process_recording, args=(text_widget, audio_data), daemon=True).start()
    else:
        text_widget.insert(tk.END, "No recording data available.\n")

# Set up the Tkinter GUI
root = tk.Tk()
root.title("PC Real-Time Speech-to-Text")

text_box = tk.Text(root, wrap="word", height=20, width=60)
text_box.pack(padx=10, pady=10)

# Add Start and Stop Recording buttons
start_button = tk.Button(root, text="Start Recording", command=lambda: start_listening(text_box))
start_button.pack(side="left", padx=10, pady=10)

stop_button = tk.Button(root, text="Stop Recording", command=lambda: stop_listening(text_box))
stop_button.pack(side="left", padx=10, pady=10)

root.mainloop()
