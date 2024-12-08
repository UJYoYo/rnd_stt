import whisper
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisperx
import os

# Check if CUDA is available and set the device to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Whisper model for Thai language with GPU support
print("Loading Whisper model...")
model = whisperx.load_model("large", device=device)

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

# Function to perform speaker diarization, transcription, and word-level alignment
def diarize_transcribe_and_align(file_path):
    print(f"Starting diarization on: {file_path}")

    # Perform speaker diarization on the audio file
    diarization = pipeline({'uri': 'filename', 'audio': file_path})

    # Full transcription results for alignment
    print("Transcribing full audio for alignment...")
    transcription_result = model.transcribe(file_path, language="th", batch_size=16)
    print(transcription_result)
    
    # Load WhisperX for word-level alignment
    print("Loading WhisperX for word-level alignment...")
    alignment_model, metadata = whisperx.load_align_model(
    language_code=transcription_result["language"],
    model_name="airesearch/wav2vec2-large-xlsr-53-th",  # Specify the Thai Wav2Vec2 model
    device=device
    )

    print("Aligning words...")
    aligned_result = whisperx.align(
        transcription_result["segments"], alignment_model, metadata, file_path, device
    )

    word_segments = aligned_result["word_segments"]
    # print(word_segments)

    # Combine speaker diarization and word-level alignment
    print("Combining Speaker Diarization with Word-Level Alignment...")
    speaker_aligned_words = []
    for word_info in word_segments:
        print(word_info)
        # print(len(word_info))
        word_text = word_info['word']
        if len(word_info) == 4:
            word_start = word_info['start']
            word_end = word_info['end']

        # Match word-level alignment with speaker diarization
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= word_start <= turn.end:  # Match word with speaker segment
                speaker_aligned_words.append({
                    "speaker": speaker,
                    "word": word_text,
                    "start": word_start,
                    "end": word_end
                })
                break

    return speaker_aligned_words

audio_file_path = "C:/Users/ChutikarnKanchanaart/Desktop/rnd/recordings/normalized.wav"
print(f"Audio file: {audio_file_path}")

# Run the diarization, transcription, and alignment
output = diarize_transcribe_and_align(audio_file_path)

# Print the results
print("\nSpeaker-Attributed Word-Level Transcription Results:")
for word_entry in output:
    print(
        f"Speaker {word_entry['speaker']}: '{word_entry['word']}' "
        f"(from {word_entry['start']:.2f}s to {word_entry['end']:.2f}s)"
    )
