# This is the code we use for whisperx with speaker diarization:
import torch
from pyannote.audio import Pipeline
import whisperx

# Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

auth_token = "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"     # Use your auth token from hugging face

# Paths
audio_path = "../recordings/nhso-nashi.wav"     # Put audio file path

output_file_path = "../results/whisper_x_transcription_1.txt"

# Load PyAnnote for speaker diarization
print("Loading PyAnnote pipeline for speaker diarization...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)

print("Performing speaker diarization...")
diarization = pipeline(audio_path)

# Load WhisperX for transcription
print("Loading WhisperX for transcription...")
whisper_model = whisperx.load_model("large", device=device)


print("Transcribing audio with WhisperX...")
transcription_result = whisper_model.transcribe(audio_path, language="th")

# Perform word-level alignment
print("Performing word-level alignment with WhisperX...")
alignment_model = whisperx.load_align_model(language_code="th", device=device)
aligned_result = whisperx.align(
    transcription_result["segments"], alignment_model, audio_path, device
)

# Combine diarization and transcription results
print("Combining diarization and transcription results...")
aligned_words = aligned_result["word_segments"]
speaker_aligned_transcripts = []

for word in aligned_words:
    for segment, speaker in diarization.itertracks(yield_label=True):
        if segment.start <= word["start"] <= segment.end:
            speaker_aligned_transcripts.append({
                "speaker": speaker,
                "word": word["text"],
                "start": word["start"],
                "end": word["end"],
            })
            break

# Print the speaker-attributed transcription
print("\n--- Speaker-Attributed Transcription ---\n")
for entry in speaker_aligned_transcripts:
    print(
        f"Speaker {entry['speaker']}: {entry['word']} "
        f"(from {entry['start']:.2f}s to {entry['end']:.2f}s)"
    )
    transcription_line = (
        f"Speaker {entry['speaker']}: {entry['word']} "
        f"(from {entry['start']:.2f}s to {entry['end']:.2f}s)"
    )
    print(transcription_line)
    output_file.write(transcription_line + "\n")


print(f"\nTranscription saved to {output_file_path}")