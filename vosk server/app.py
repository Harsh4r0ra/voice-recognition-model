from flask import Flask, render_template, jsonify, Response
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import queue
import json
import threading
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for audio handling
q = queue.Queue()
model = Model(lang="en-us")
recognizer = None
recording = False

def audio_callback(indata, frames, time, status):
    """Callback for audio stream"""
    if status:
        print(status)
    q.put(bytes(indata))

def process_audio():
    """Process audio data and emit results via Socket.IO"""
    global recording
    
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                             channels=1, callback=audio_callback):
            print("Recording started")
            
            while recording:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result["text"]:
                        socketio.emit('transcript', {'text': result["text"], 'final': True})
                else:
                    partial = json.loads(recognizer.PartialResult())
                    if partial["partial"]:
                        socketio.emit('transcript', {'text': partial["partial"], 'final': False})
                        
    except Exception as e:
        print(f"Error in audio processing: {e}")
        socketio.emit('error', {'message': str(e)})

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected')

@socketio.on('start_recording')
def handle_start_recording():
    global recording, recognizer
    
    if not recording:
        recording = True
        recognizer = KaldiRecognizer(model, 16000)
        threading.Thread(target=process_audio).start()
        emit('recording_started')

@socketio.on('stop_recording')
def handle_stop_recording():
    global recording
    
    if recording:
        recording = False
        emit('recording_stopped')

@socketio.on('disconnect')
def handle_disconnect():
    global recording
    recording = False
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)