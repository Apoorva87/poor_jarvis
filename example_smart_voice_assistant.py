import sys
import sounddevice as sd
import asyncio
from src.tts.coqui_agent import CoquiTTSAgent
from src.stt.whisper_agent import WhisperSTTAgent
from src.vad.processor import VADProcessor
from src.llm.ollama_agent import OllamaAgent
from src.llm.openai_agent import OpenAIAgent

async def main():
    # Initialize agents
    tts_agent = CoquiTTSAgent()
    stt_agent = WhisperSTTAgent()
    llm_agent = OllamaAgent()
    #llm_agent = OpenAIAgent()
    vad_processor = VADProcessor()
    
    print("\n[SYSTEM] Ready! Listening for your voice...")
    print("Press Ctrl+C to stop.")

    try:
        with sd.InputStream(samplerate=vad_processor.sample_rate, channels=1, dtype='int16') as stream:
            frames = vad_processor.audio_frame_generator(stream)
            
            for audio_segment in vad_processor.vad_collector(frames):
                if audio_segment:
                    print(f"\n[SYSTEM] Detected speech segment ({len(audio_segment) / (2 * vad_processor.sample_rate):.2f} seconds).")
                    transcribed_text = await stt_agent.transcribe(audio_segment)

                    if transcribed_text.strip().lower() in ["exit", "quit", "stop"]:
                        print("Exiting.")
                        break
                  
                    print(f"Passing this down to llm : {transcribed_text}")
                    if transcribed_text:
                        llm_response = await llm_agent.get_response(transcribed_text)
                        if llm_response:
                            await tts_agent.synthesize(llm_response)
                    else:
                        print("[SYSTEM] No clear speech detected for transcription.")
                else:
                    print("[SYSTEM] No audio segment captured by VAD.")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping conversational AI system.")
    except Exception as e:
        print(f"\n[SYSTEM] An unexpected error occurred: {e}", file=sys.stderr)
        print("[SYSTEM] Please ensure your microphone is connected and working.", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())


