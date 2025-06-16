#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys
import re

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, InputImageRawFrame, OutputImageRawFrame, TextFrame, TransportMessageUrgentFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.services.piper.tts import PiperTTSService


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class EdgeDetectionProcessor(FrameProcessor):
    def __init__(self, camera_out_width, camera_out_height: int):
        super().__init__()
        self._camera_out_width = camera_out_width
        self._camera_out_height = camera_out_height

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputImageRawFrame):
            # Convert bytes to NumPy array
            img = np.frombuffer(frame.image, dtype=np.uint8).reshape(
                (frame.size[1], frame.size[0], 3)
            )

            # perform edge detection
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # convert the size if needed
            desired_size = (self._camera_out_width, self._camera_out_height)
            if frame.size != desired_size:
                resized_image = cv2.resize(img, desired_size)
                frame = OutputImageRawFrame(resized_image.tobytes(), desired_size, frame.format)
                await self.push_frame(frame)
            else:
                await self.push_frame(
                    OutputImageRawFrame(image=img.tobytes(), size=frame.size, format=frame.format)
                )
        else:
            await self.push_frame(frame, direction)


class LoggingProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Log only text frames (i.e., STT output)
        if isinstance(frame, TextFrame):
            logger.info(f"STT Output before LLM: {frame.text}")
        await self.push_frame(frame, direction)


class ResponseCleaningProcessor(FrameProcessor):
    """Processor to clean LLM responses by removing <think></think> tags."""
    
    def __init__(self):
        super().__init__()
    
    def clean_response(self, text):
        """Remove <think> tags and emoticons from the response."""
        if not text:
            return text
        
        # Remove complete <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove incomplete <think> tags (when response is cut off)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
        
        # Remove emoticons and emojis
        # Unicode emoji ranges (covers most common emojis)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+", flags=re.UNICODE)
        
        cleaned = emoji_pattern.sub('', cleaned)
        
        # Remove common text-based emoticons
        text_emoticons = [
            r':\)', r':-\)', r':\(', r':-\(', r':D', r':-D', r':P', r':-P',
            r':o', r':-o', r':O', r':-O', r';\)', r';-\)', r':\|', r':-\|',
            r':/', r':-/', r':\\', r':-\\', r':s', r':-s', r':S', r':-S',
            r'<3', r'</3', r'=\)', r'=D', r'=P', r'=\(', r'>\.<', r'T_T',
            r'T\.T', r'-_-', r'\^_\^', r'\^-\^', r'o_o', r'O_O', r'@_@',
            r'x_x', r'X_X', r'>:\(', r'>:-\(', r'D:<', r'D-:<'
        ]
        
        for emoticon in text_emoticons:
            cleaned = re.sub(emoticon, '', cleaned)
        
        # Clean up extra whitespace and newlines
        cleaned = re.sub(r'\n\s*\n+', '\n', cleaned.strip())
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned)
        
        return cleaned
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        logger.info(f"Local Processing frame: {frame}")
   
        if (isinstance(frame, TextFrame) ) and direction == FrameDirection.DOWNSTREAM:
            original_text = frame.text
            cleaned_text = self.clean_response(original_text)
            
            # Log the cleaning if there was a change
            if original_text != cleaned_text:
                logger.info(f"🧹 Cleaned LLM response:")
                logger.info(f"   Original: {original_text}")
                logger.info(f"   Cleaned:  {cleaned_text}")
            
            # Create new frame with cleaned text
            cleaned_frame = TextFrame(cleaned_text)
            await self.push_frame(cleaned_frame, direction)
        else:
            # Pass through all other frames unchanged
            await self.push_frame(frame, direction)
        

SYSTEM_INSTRUCTION = f"""
"You are Gemini Chatbot, a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""


async def run_bot(webrtc_connection):
    transport_params = TransportParams(
        #camera_in_enabled=True,
        #camera_out_enabled=True,
        #camera_out_is_live=True,
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        vad_audio_passthrough=True,
        audio_out_10ms_chunks=2,
    )

    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection, params=transport_params
    )

    # OllamaLLMService
    llm_input_params = OpenAILLMService.InputParams()
    llm_input_params.extra={'extra_query':{'think':False}}

    llm = OpenAILLMService(
        model="qwen3:8b",
        api_key="ollama",
        stream=False,
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        params=llm_input_params
    )
  
        

    # https://docs.pipecat.ai/server/services/llm/ollama#constructor-parameters

    from pipecat.services.whisper.stt import WhisperSTTService, Model

    # Configure service with default model
    stt = WhisperSTTService(
        model=Model.DISTIL_MEDIUM_EN,
        device="cpu",
        no_speech_prob=0.4
    )

    context = OpenAILLMContext(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant. Do not generate <think> tags. /no_think",
            }
        ],
    )
    context_aggregator = llm.create_context_aggregator(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Response cleaning processor
    response_cleaner = ResponseCleaningProcessor()

    # Add Piper TTS service (free, fast, lightweight)
    #tts = PiperTTSService(
    #    voice="en_US-lessac-medium",  # High-quality English voice
    #    # voice="en_US-amy-medium",   # Alternative voice option
    #    # voice="en_US-ryan-medium",  # Male voice option
    #)

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            #LoggingProcessor(),  # <-- Logging between STT and LLM
            llm,  # LLM
            #response_cleaner,  # <-- Clean LLM output before sending to WebRTC
            #tts,  # <-- Add TTS here (after response cleaning)
            #EdgeDetectionProcessor(
            #    transport_params.camera_out_width, transport_params.camera_out_height
            #),  # Sending the video back to the user
            pipecat_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            observers=[RTVIObserver(rtvi)],
        ),
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")

    @pipecat_transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info("Pipecat Client closed")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
