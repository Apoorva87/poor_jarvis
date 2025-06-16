#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys
import asyncio
import json
import numpy as np
from typing import Optional
import threading
import queue
import time
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.llm.ollama_agent import OllamaAgent
from src.stt.whisper_agent import WhisperSTTAgent
from src.vad.processor import VADProcessor
from src.tts.coqui_agent import CoquiTTSAgent

# Pipecat imports for minimal pipeline
from pipecat.frames.frames import Frame, AudioRawFrame, LLMTextFrame, TranscriptionFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class DirectAudioProcessor(FrameProcessor):
    """
    A minimal frame processor that handles audio with src components directly.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize src components
        self.vad_processor = VADProcessor(
            sample_rate=16000,
            frame_duration_ms=30,
            padding_duration_ms=500,
            vad_mode=3
        )
        self.stt_agent = WhisperSTTAgent(model_name="base.en")
        self.llm_agent = OllamaAgent(model_name="qwen3:8b", max_history=10)
        self.tts_agent = CoquiTTSAgent()
        
        # Audio processing
        self.audio_buffer = bytearray()
        self.frame_size = self.vad_processor.frame_size * 2  # 2 bytes per int16 sample
        
        # VAD state
        self.triggered = False
        self.voiced_frames = []
        self.ring_buffer = []
        self.ring_buffer_maxlen = int(500 / 30)  # padding frames
        
        # Communication queues
        self.text_queue = queue.Queue(maxsize=5)
        
        # Threading
        self.processing_thread = None
        self.tts_active = threading.Event()
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        """Process frames - handle audio input and text output."""
        
        # Always pass through pipeline control frames first
        #if hasattr(frame, '__class__') and frame.__class__.__name__ in ['StartFrame', 'EndFrame', 'CancelFrame']:
        #   await self.push_frame(frame, direction)
        #
        # return
        
        if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            # Process incoming audio with VAD
            if not self.tts_active.is_set():
                await self._process_audio_frame(frame)
            # Don't pass audio frames downstream to avoid echo
            return
        
        elif isinstance(frame, TextFrame) and direction == FrameDirection.UPSTREAM:
            # This is a response from our LLM - pass it through for TTS
            await self.push_frame(frame, direction)
            return
        
        # Pass through all other frames
        await self.push_frame(frame, direction)
    
    async def _process_audio_frame(self, frame: AudioRawFrame):
        """Process incoming audio frame with VAD."""
        try:
            # Get audio data from AudioRawFrame
            audio_data = frame.audio
            self.audio_buffer.extend(audio_data)
            
            # Process complete frames
            while len(self.audio_buffer) >= self.frame_size:
                frame_bytes = bytes(self.audio_buffer[:self.frame_size])
                self.audio_buffer = self.audio_buffer[self.frame_size:]
                
                # Check if this frame contains speech
                try:
                    is_speech = self.vad_processor.is_speech(frame_bytes)
                except Exception as e:
                    logger.error(f"VAD error: {e}")
                    continue
                
                # VAD logic (similar to example_parallel_sva.py)
                if not self.triggered:
                    self.ring_buffer.append((frame_bytes, is_speech))
                    if len(self.ring_buffer) > self.ring_buffer_maxlen:
                        self.ring_buffer.pop(0)
                    
                    if is_speech:
                        self.triggered = True
                        logger.info("[VAD] Speech detected. Starting recording...")
                        for f, s in self.ring_buffer:
                            self.voiced_frames.append(f)
                        self.ring_buffer.clear()
                else:
                    self.voiced_frames.append(frame_bytes)
                    self.ring_buffer.append((frame_bytes, is_speech))
                    if len(self.ring_buffer) > self.ring_buffer_maxlen:
                        self.ring_buffer.pop(0)
                    
                    if not is_speech:
                        num_unvoiced = len([f for f, s in self.ring_buffer if not s])
                        if num_unvoiced == self.ring_buffer_maxlen:
                            self.triggered = False
                            # Remove padding frames
                            for _ in range(self.ring_buffer_maxlen):
                                if self.voiced_frames:
                                    self.voiced_frames.pop()
                            
                            logger.info("[VAD] Speech segment ended. Processing...")
                            
                            # Send complete audio segment for STT
                            if self.voiced_frames:
                                complete_audio = b''.join(self.voiced_frames)
                                await self._process_speech_segment(complete_audio)
                            
                            self.voiced_frames = []
                            self.ring_buffer.clear()
                            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
    
    async def _process_speech_segment(self, audio_data: bytes):
        """Process a complete speech segment with STT."""
        try:
            logger.info(f"[STT] Processing audio segment ({len(audio_data)} bytes)")
            
            # Transcribe with Whisper
            transcription = await self.stt_agent.transcribe(audio_data)
            
            if transcription and transcription.strip():
                logger.info(f"[STT] Transcription: '{transcription}'")
                
                # Send user transcript to frontend via TranscriptionFrame
                transcript_frame = TranscriptionFrame(
                    text=transcription.strip(),
                    user_id="user",
                    timestamp=datetime.now().isoformat()
                )
                await self.push_frame(transcript_frame, FrameDirection.UPSTREAM)
                
                # Add to processing queue for LLM
                try:
                    self.text_queue.put({
                        'text': transcription.strip(),
                        'timestamp': time.time()
                    }, timeout=0.1)
                except queue.Full:
                    logger.warning("Text queue full, dropping message")
            else:
                logger.info("[STT] No transcription result")
                
        except Exception as e:
            logger.error(f"[STT] Error during transcription: {e}")
    
    def _processing_loop(self):
        """Background thread for LLM processing and TTS."""
        logger.info("[PROCESSING] Starting LLM processing loop...")
        
        while self.running:
            try:
                # Get text from queue
                try:
                    text_data = self.text_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                user_input = text_data['text']
                logger.info(f"[LLM] Processing: '{user_input}'")
                
                # Get LLM response (this is sync, so we run it in thread)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    response = loop.run_until_complete(
                        self.llm_agent.get_response(user_input)
                    )
                    
                    if response and response.strip():
                        logger.info(f"[LLM] Generated response: '{response}'")
                        
                        # Create TextFrame and push it upstream for TTS and transcript handling
                        text_frame = LLMTextFrame(response.strip())
                        loop.run_until_complete(
                            self.push_frame(text_frame, FrameDirection.UPSTREAM)
                        )
                        
                        # Set TTS active flag during synthesis
                        self.tts_active.set()
                        
                        # TTS synthesis and playback (local audio output)
                        loop.run_until_complete(
                            self.tts_agent.synthesize(response.strip())
                        )
                        
                        # Clear TTS active flag
                        self.tts_active.clear()
                        
                    else:
                        logger.info("[LLM] No response generated")
                        
                except Exception as e:
                    logger.error(f"[LLM] Error: {e}")
                    self.tts_active.clear()
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(f"[PROCESSING] Error in processing loop: {e}")
                
        logger.info("[PROCESSING] Processing loop ended")
    
async def run_bot(webrtc_connection):
    """Main bot function using minimal pipeline with direct src integration."""
    
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=False,  # We'll use our custom VAD
        vad_audio_passthrough=True,
    )

    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection, params=transport_params
    )

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Create our direct audio processor
    audio_processor = DirectAudioProcessor()

    # Create minimal pipeline with RTVI
    pipeline = Pipeline([
        pipecat_transport.input(),
        rtvi,
        audio_processor,
        pipecat_transport.output(),
    ])

    task = PipelineTask(
        pipeline, 
        params=PipelineParams(
            allow_interruptions=True,
            observers=[RTVIObserver(rtvi)]
        )
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("RTVI client ready")
        await rtvi.set_bot_ready()

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Send initial greeting
        greeting_frame = TextFrame("Hello! I'm your voice assistant. How can I help you today?")
        await task.queue_frames([greeting_frame])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")

    @pipecat_transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info("Client closed")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task) 