import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
from tempfile import NamedTemporaryFile

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate),
                      samplerate=sample_rate,
                      channels=1)
    sd.wait()
    return recording

def transcribe_audio(audio_array, model):
    """Transcribe audio using Whisper model"""
    # Save the numpy array as a temporary WAV file
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        # Explicitly specify the temporary file path and make sure the file is written before processing
        temp_file_path = temp_file.name
        sf.write(temp_file_path, audio_array, 16000)
        
        # Transcribe the audio file
        result = model.transcribe(temp_file_path)
        return result["text"]

def main():
    # Load the Whisper model (you can choose different sizes: tiny, base, small, medium, large)
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    while True:
        input("Press Enter to start recording (or Ctrl+C to exit)...")
        
        # Record audio
        audio = record_audio()
        print("Recording finished!")
        
        # Transcribe
        print("Transcribing...")
        text = transcribe_audio(audio, model)
        
        print("\nTranscription:")
        print(text)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
