

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