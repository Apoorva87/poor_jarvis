import sys
import sounddevice as sd
import asyncio
import numpy as np
from src.tts.coqui_agent import CoquiTTSAgent
from src.stt.whisper_agent import WhisperSTTAgent
from src.vad.processor import VADProcessor
from src.llm.ollama_agent import OllamaAgent
from src.llm.openai_agent import OpenAIAgent
import multiprocessing as mp
from multiprocessing import Queue, Process, Event


class VoiceAssistantMultiprocess:
    def __init__(self):
        # Inter-process communication
        self.audio_queue = mp.Queue(maxsize=10)
        self.text_queue = mp.Queue(maxsize=5)
        self.interrupt_event = mp.Event()
        self.tts_active_event = mp.Event()  # New event for TTS state

    def start(self):
        print("[SYSTEM] Available audio devices:")
        print(sd.query_devices())
        print("\n[SYSTEM] Default input device:", sd.query_devices(kind='input'))
        print("[SYSTEM] Default output device:", sd.query_devices(kind='output'))
        
        # Process 1: Audio I/O (capture + playback) + STT
        audio_process = Process(target=self.audio_io_process)

        # Process 2:  + LLM (CPU intensive) + TTS
        processing_process = Process(target=self.processing_process)

        processes = [audio_process, processing_process]

        for p in processes:
            p.start()

        return processes

    def audio_io_process(self):
        """Single process handles both capture and playback"""
        # Avoids audio device conflicts
        asyncio.run(self.audio_io_main())

    async def audio_io_main(self):
        # Run VAD capture and TTS playback concurrently in same process
        await asyncio.gather(
            self.capture_with_vad(),
        )

    async def capture_with_vad(self):
        stt_agent = WhisperSTTAgent()
        vad_processor = VADProcessor()
        
        print("[AUDIO] Starting audio capture with VAD...")
        print(f"[AUDIO] Sample rate: {vad_processor.sample_rate}")
        print(f"[AUDIO] Using device: {sd.query_devices(kind='input')}")
        
        try:
            with sd.InputStream(samplerate=vad_processor.sample_rate, channels=1, dtype='int16') as stream:
                print("[AUDIO] Successfully opened audio input stream")
                frames = vad_processor.audio_frame_generator(stream)
                
                for audio_segment in vad_processor.vad_collector(frames):
                    # Skip VAD processing if TTS is active
                    if self.tts_active_event.is_set():
                        continue
                        
                    if audio_segment:
                        print(f"\n[AUDIO] Detected speech segment ({len(audio_segment) / (2 * vad_processor.sample_rate):.2f} seconds).")
                       
                        transcribed_text = await stt_agent.transcribe(audio_segment)

                        if transcribed_text.strip().lower() in ["exit", "quit", "stop"]:
                            print("Exiting.")
                            self.interrupt_event.set()
                            break

                        if transcribed_text:
                            # Add transcribed text to processing queue
                            await self.add_to_text_queue(transcribed_text)
                    
                    # Check if we should stop
                    if self.interrupt_event.is_set():
                        break
                        
        except Exception as e:
            print(f"[AUDIO] Error in audio capture: {e}")
            print("[AUDIO] Full error details:", sys.exc_info())

    async def add_to_text_queue(self, text):
        """Add transcribed text to processing queue"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._put_with_timeout,
                self.text_queue,
                {
                    'text': text,
                    'timestamp': loop.time()
                },
                0.1
            )
            print(f"[AUDIO] Added text to queue: '{text}'")
        except Exception as e:
            print(f"[AUDIO] Error adding to queue: {e}")

    def _put_with_timeout(self, queue, item, timeout):
        """Helper method to put item in queue with timeout"""
        try:
            queue.put(item, timeout=timeout)
        except:
            print("[AUDIO] Warning: Text queue full, dropping message")
 
    def processing_process(self):
        """Dedicated process for STT + LLM"""
        asyncio.run(self.processing_main())

    async def processing_main(self):
        print("[PROCESSING] Starting LLM + TTS processing...")
        tts_agent = CoquiTTSAgent()
        llm_agent = OllamaAgent()
        
        while True:
            # Get transcribed text from queue
            text_data = await self.get_from_text_queue()
            
            if text_data:
                transcribed_text = text_data['text']
                print(f"[PROCESSING] Processing: '{transcribed_text}'")
                
                try:
                    # LLM processing
                    response = await llm_agent.get_response(transcribed_text)
                    
                    if response:
                        # Set TTS active flag
                        self.tts_active_event.set()
                        
                        # TTS synthesis and playback
                        await tts_agent.synthesize(response)
                        
                        # Clear TTS active flag
                        self.tts_active_event.clear()
                        
                except Exception as e:
                    print(f"[PROCESSING] Error: {e}")
                    self.tts_active_event.clear()  # Ensure flag is cleared on error
            
            # Check if we should stop
            if self.interrupt_event.is_set():
                break
                
            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning

    async def get_from_text_queue(self):
        """Get text from queue"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._get_with_timeout,
                self.text_queue,
                1.0
            )
        except Exception as e:
            print(f"[PROCESSING] Error getting from queue: {e}")
            return None

    def _get_with_timeout(self, queue, timeout):
        """Helper method to get item from queue with timeout"""
        try:
            return queue.get(timeout=timeout)
        except:
            return None

if __name__ == "__main__":
    print("Starting Voice Assistant with Multiprocessing...")
    vamp = VoiceAssistantMultiprocess()
    processes = vamp.start()
    
    try:
        # Wait for all processes to complete
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping Voice Assistant...")
        vamp.interrupt_event.set()
        for p in processes:
            p.terminate()
            p.join()


