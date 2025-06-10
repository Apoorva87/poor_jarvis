import torch
from TTS.api import TTS
import sounddevice as sd

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# List available 🐸TTS models
print(TTS().list_models())


model="tts_models/en/vctk/vits"
tts = TTS(model).to(device)
for i in range(225, 235):
    wav = tts.tts(text=f"I am Apoorva, I like play tik tok and mahjong. Running on vits. Person number p{i}", speaker=f'p{i}')
    sd.play(wav, samplerate=24000)
    sd.wait()


# Init TTS
model_name = "tts_models/en/ljspeech/tacotron2-DDC"
tts = TTS(model_name).to(device)

# model = AutoModelForCausalLM.from_pretrained(
#    "model_name_or_path",
#    trust_remote_code=True
#)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="I am Apoorva. I like play tik tok and mahjong ")
sd.play(wav, samplerate=24000)
sd.wait()
wav = tts.tts(text="I am watch Raj Shamani and I can tell you he is not tthat good.")
sd.play(wav, samplerate=24000)
sd.wait()

# Text to speech to a file
#tts.tts_to_file(text="Hello world!", speaker_wav="speaker.wav", language="en", file_path="output.wav")
model="tts_models/en/ljspeech/fast_pitch"
tts = TTS(model).to(device)

wav = tts.tts(text="I am Apoorva, I like play tik tok and mahjong. Running fast_pitch")
sd.play(wav, samplerate=24000)
sd.wait()
wav = tts.tts(text="I am watch Raj Shamani and I can tell you he is not tthat good.")
sd.play(wav, samplerate=24000)
sd.wait()

model="tts_models/en/ljspeech/tacotron2-DDC_ph"
tts = TTS(model).to(device)

wav = tts.tts(text="I am Apoorva, I like play tik tok and mahjong. Running tachoDDC2")
sd.play(wav, samplerate=24000)
sd.wait()
wav = tts.tts(text="I am watch Raj Shamani and I can tell you he is not tthat good.")
sd.play(wav, samplerate=24000)
sd.wait()

model="tts_models/multilingual/multi-dataset/your_tts"
tts = TTS(model).to(device)

wav = tts.tts(text="I am Apoorva, I like play tik tok and mahjong. Running on your tts")
sd.play(wav, samplerate=24000)
sd.wait()
wav = tts.tts(text="I am watch Raj Shamani and I can tell you he is not tthat good.")
sd.play(wav, samplerate=24000)
sd.wait()





