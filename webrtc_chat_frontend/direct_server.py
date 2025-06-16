#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from direct_webrtc_bot import run_bot  # Import from our direct bot
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import sys
import os

from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

# Load environment variables
load_dotenv(override=True)

# Define possible paths to the dist directory
base_dir = os.path.dirname(__file__)
possible_dist_paths = [
    os.path.abspath(os.path.join(base_dir, "client", "dist")), # in prod
    os.path.abspath(os.path.join(base_dir, "..", "client", "dist")),  # in dev
]

dist_dir = None

# Try each possible path
for path in possible_dist_paths:
    print(f"Looking for dist directory at: {path}")
    logging.info(f"Checking dist directory path: {path}")
    if os.path.isdir(path):
        dist_dir = path
        break

if not dist_dir:
    logging.error("Static frontend build not found in any of the expected locations.")
    raise RuntimeError(
        "Static frontend build not found. Please run `npm run build` in the client directory."
    )

SmallWebRTCPrebuiltUI = StaticFiles(directory=dist_dir, html=True)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("direct-webrtc-server")

app = FastAPI()

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = ["stun:stun.l.google.com:19302"]

# Mount the frontend at /
app.mount("/client", SmallWebRTCPrebuiltUI)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"], type=request["type"], restart_pc=request.get("restart_pc", False)
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # This creates the connection to the direct bot (no Pipecat pipeline)
        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.close() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct WebRTC server with src components")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7862, help="Port for HTTP server (default: 7862)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print("🚀 Starting Direct WebRTC Voice Chat Server...")
    print(f"📡 Server will run on http://{args.host}:{args.port}")
    print("🎯 Direct integration with src components (NO Pipecat pipeline):")
    print("   • VAD: src/vad/processor.py (WebRTC VAD)")
    print("   • STT: src/stt/whisper_agent.py (Whisper)")
    print("   • LLM: src/llm/ollama_agent.py (Ollama)")
    print("   • TTS: src/tts/coqui_agent.py (Coqui)")
    print("🌐 Frontend: voice_chat_frontend WebRTC client")
    print("⚡ Audio processing: Direct byte-level processing (like example_parallel_sva.py)")
    
    uvicorn.run(app, host=args.host, port=args.port) 