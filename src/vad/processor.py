import collections
import webrtcvad
import numpy as np
import sounddevice as sd
import sys

class VADProcessor:
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30, 
                 padding_duration_ms: int = 500, vad_mode: int = 3):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.vad = webrtcvad.Vad(vad_mode)

    def audio_frame_generator(self, stream):
        """Generates audio frames from the input stream."""
        while True:
            try:
                data, overflowed = stream.read(self.frame_size)
                if overflowed:
                    print("Warning: Audio input buffer overflowed!", file=sys.stderr)
                yield data.astype(np.int16).tobytes()
            except Exception as e:
                print(f"Error reading from audio stream: {e}", file=sys.stderr)
                break

    def is_speech(self, frame):
        return self.vad.is_speech(frame, self.sample_rate)

    def vad_collector(self, frames):
        """Filters out non-speech frames using VAD and collects speech segments."""
        num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        
        for frame in frames:
            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                if is_speech:
                    triggered = True
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
                    print(f"\n[VAD] Speech detected. Starting recording...")
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                if not is_speech:
                    num_unvoiced = len([f for f, s in ring_buffer if not s])
                    if num_unvoiced == ring_buffer.maxlen:
                        triggered = False
                        for _ in range(ring_buffer.maxlen):
                            voiced_frames.pop()
                        print("\n[VAD] Speech segment ended. Processing...")
                        yield b''.join(voiced_frames)
                        voiced_frames = []
                        ring_buffer.clear()
        
        if voiced_frames:
            print("\n[VAD] Stream ended while speech was active. Processing last segment...")
            yield b''.join(voiced_frames) 
