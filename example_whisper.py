import whisper

model = whisper.load_model("tiny.en")
#model = whisper.load_model("base.en")
result = model.transcribe("speaker.wav")
print(result["text"])
