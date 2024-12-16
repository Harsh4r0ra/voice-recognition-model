import whisper
import sounddevice as sd
import numpy as np
import torch
import argparse
import subprocess
import sys
import threading
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        print("Error: FFmpeg required but not found.")
        print(f"Install command: {'brew install ffmpeg' if sys.platform == 'darwin' else 'sudo apt-get install ffmpeg' if sys.platform == 'linux' else 'Download from https://www.gyan.dev/ffmpeg/builds/'}")
        return False

class AudioTranscriber:
    def __init__(self, model_size: str = "base", sample_rate: int = 16000):
        if not check_ffmpeg():
            raise RuntimeError("FFmpeg is required but not found.")
            
        self.sample_rate = sample_rate
        self.model = whisper.load_model(model_size).cpu()
        self.recording_data = []
        self.stop_recording = False
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if status:
            print(f"\nWarning: {status}")
        self.recording_data.extend(indata.copy())

    def record_audio(self) -> np.ndarray:
        """Record audio from the microphone until user stops."""
        try:
            self.recording_data = []
            self.stop_recording = False
            
            stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback
            )
            
            print("Recording... Press Enter to stop")
            
            with stream:
                while not self.stop_recording:
                    time.sleep(0.1)
            
            # Convert recorded data to numpy array and normalize
            audio_data = np.concatenate(self.recording_data)
            audio_data = audio_data.flatten()  # Ensure 1D array
            
            # Normalize audio to float32 range [-1, 1]
            audio_data = audio_data.astype(np.float32)
            if audio_data.size > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data
            
        except Exception as e:
            print(f"Recording failed: {str(e)}")
            return np.array([], dtype=np.float32)

    def transcribe_audio(self, audio_array: np.ndarray) -> str:
        """Transcribe audio using the Whisper model."""
        if audio_array.size == 0:
            return ""
            
        try:
            print("Transcribing...", end='', flush=True)
            result = self.model.transcribe(audio_array, fp16=False)
            print("\r", end='')  # Clear transcribing message
            return result["text"].strip()
                
        except Exception as e:
            print(f"Transcription failed: {str(e)}")
            return ""

def input_thread(transcriber):
    """Thread to handle input"""
    input()
    transcriber.stop_recording = True

def main():
    if not check_ffmpeg():
        return

    parser = argparse.ArgumentParser(description='Real-time Audio Transcription using Whisper')
    parser.add_argument('--model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Model size')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Sample rate (Hz)')
    
    args = parser.parse_args()
    
    try:
        transcriber = AudioTranscriber(args.model, args.sample_rate)
        
        while True:
            input("Press Enter to start recording (Ctrl+C to exit)")
            
            # Start input thread
            input_thread_handle = threading.Thread(
                target=input_thread, 
                args=(transcriber,),
                daemon=True
            )
            input_thread_handle.start()
            
            # Record audio
            audio = transcriber.record_audio()
            
            # Only transcribe if we have audio data
            if audio.size > 0:
                text = transcriber.transcribe_audio(audio)
                if text:
                    print(f"Transcript: {text}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()