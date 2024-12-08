import whisper
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os

# Check if CUDA is available and set the device to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Whisper model for Thai language with GPU support
print("Loading Whisper model...")
model = whisper.load_model("large", device=device)

use_auth_token = "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"

try:
    # Load the pre-trained speaker diarization pipeline
    print("Loading speaker diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=use_auth_token
    )
    pipeline = pipeline.to(torch.device(device))
except Exception as e:
    print(f"Failed to load the pipeline: {e}")
    os.sys.exit(1)

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    print(f"Transcribing segment: {file_path}")
    result = model.transcribe(file_path, language="th")
    print(result)
    return result['text']

# Function to perform speaker diarization and transcribe
def diarize_and_transcribe(file_path):
    print(f"Starting diarization on: {file_path}")

    # Perform speaker diarization on the audio file
    diarization = pipeline({'uri': 'filename', 'audio': file_path})

    segments = []
    audio_segment = AudioSegment.from_file(file_path, format="wav")  # Input is WAV


    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        print(f"Processing segment: Speaker {speaker}, Start: {start_time}, End: {end_time}")

        # Extract the segment directly from the WAV file without creating a temp file
        segment_path = f"segment_{start_time:.2f}_{end_time:.2f}.wav"
        segment = audio_segment[start_time * 1000:end_time * 1000]  # Convert seconds to milliseconds
        segment.export(segment_path, format="wav")

        # Transcribe the segment
        transcribed_text = transcribe_audio(segment_path)
        segments.append((speaker, transcribed_text))

        # Clean up the segment file after processing
        if os.path.exists(segment_path):
            os.remove(segment_path)

    return segments

audio_file_path = "C:/Users/ChutikarnKanchanaart/Desktop/rnd/recordings/nhso-nashi.wav"
print(f"Audio file: {audio_file_path}")

# Run the diarization and transcription
output = diarize_and_transcribe(audio_file_path)

# Print the results
print("\nDiarization and transcription results:")
for speaker, text in output:
    print(f"Speaker {speaker}: {text}")
