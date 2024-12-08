import deepcut
import jiwer

# File paths
original_text_file = "manual_transcription.txt"   # Path to the file containing the original text
transcribed_text_file = "whisper_x_transcription.txt"   # Path to the file containing the transcribed text

# Read the original and transcribed texts from the files
with open(original_text_file, "r", encoding="utf-8") as file:
    original_text = file.read().strip()

with open(transcribed_text_file, "r", encoding="utf-8") as file:
    transcribed_text = file.read().strip()

# Tokenize both the original and transcribed texts using DeepCut
original_tokenized = deepcut.tokenize(original_text)
transcribed_tokenized = deepcut.tokenize(transcribed_text)

# Join the tokenized words back into space-separated strings
original_sentence = ' '.join(original_tokenized)
transcribed_sentence = ' '.join(transcribed_tokenized)

print(f"Original Sentence: {original_sentence}")
print(f"Transcribed Sentence: {transcribed_sentence}")

# Compute the Word Error Rate (WER) using jiwer
print("Calculating WER...")
wer = jiwer.wer(original_sentence, transcribed_sentence)
print(f"WER: {wer:.2%}")
