# Direct WebRTC Voice Chat with src Components

This is a **direct integration** approach that uses the WebRTC frontend from `voice_chat_frontend` but processes audio **directly** with your `src` components, **completely bypassing the Pipecat pipeline**.

## 🎯 **Key Differences**

### **This Approach (Direct)**
- ✅ Uses your `src` components **exactly as they are**
- ✅ **No Pipecat pipeline** - direct audio processing
- ✅ Same audio processing logic as `example_parallel_sva.py`
- ✅ WebRTC frontend for browser compatibility
- ✅ Direct byte-level audio processing

### **Previous Approach (Pipeline)**
- ❌ Required adapting `src` components to Pipecat frames
- ❌ Used Pipecat pipeline (which you wanted to avoid)
- ❌ Complex frame processors and custom frame types

## 🏗️ **Architecture**

```
Browser (WebRTC Client)
    ↓ Raw Audio Bytes
Direct WebRTC Connection
    ↓ Audio Frames
DirectWebRTCBot
    ↓ Byte Processing
src/vad/processor.py (WebRTC VAD)
    ↓ Audio Segments  
src/stt/whisper_agent.py (Whisper STT)
    ↓ Text
src/llm/ollama_agent.py (Ollama LLM)
    ↓ Response Text
src/tts/coqui_agent.py (Coqui TTS)
    ↓ Audio Output
Browser (WebRTC Client)
```

## 📁 **Files**

- `direct_server.py` - Main server (port 7862)
- `direct_webrtc_bot.py` - Direct bot logic (no pipeline)
- `README_direct_integration.md` - This documentation

## 🚀 **Quick Start**

### **1. Prerequisites**
```bash
# Ensure all services are running
ollama serve

# Ensure you have the required model
ollama pull qwen3:8b

# Build the frontend (if not already done)
cd voice_chat_frontend/client
npm install
npm run build
cd ..
```

### **2. Start the Direct Server**
```bash
# From the voice_chat_frontend directory
python direct_server.py --host localhost --port 7862
```

### **3. Access the Application**
Open your browser and go to: `http://localhost:7862`

## 🔧 **How It Works**

### **Audio Processing Flow**
1. **WebRTC receives raw audio** from browser
2. **DirectWebRTCBot** processes audio bytes directly
3. **VAD** (from `src/vad/processor.py`) detects speech segments
4. **STT** (from `src/stt/whisper_agent.py`) transcribes audio
5. **LLM** (from `src/llm/ollama_agent.py`) generates response
6. **TTS** (from `src/tts/coqui_agent.py`) synthesizes speech
7. **WebRTC sends audio** back to browser

### **Threading Model**
- **Main Thread**: WebRTC connection and audio frame handling
- **Background Thread**: LLM processing and TTS synthesis
- **Queue-based Communication**: Similar to `example_parallel_sva.py`

### **VAD Logic**
Uses the **exact same VAD logic** as `example_parallel_sva.py`:
- Ring buffer for padding
- Speech detection triggers
- Complete audio segment collection

## 🔧 **Configuration**

### **Model Settings**
Edit `direct_webrtc_bot.py`:
```python
# In DirectWebRTCBot.__init__()
self.stt_agent = WhisperSTTAgent(model_name="base.en")  # STT model
self.llm_agent = OllamaAgent(model_name="qwen3:8b", max_history=10)  # LLM model
```

### **VAD Sensitivity**
```python
# In DirectWebRTCBot.__init__()
self.vad_processor = VADProcessor(
    sample_rate=16000,
    frame_duration_ms=30,
    padding_duration_ms=500,  # Adjust padding
    vad_mode=3  # 0-3, higher = more aggressive
)
```

### **Audio Processing**
```python
# Frame size calculation
self.frame_size = self.vad_processor.frame_size * 2  # 2 bytes per int16 sample
```

## 🆚 **Comparison with Other Approaches**

| Feature | Direct (`direct_server.py`) | Pipeline (`server_with_src.py`) | Original (`server.py`) |
|---------|---------------------------|--------------------------------|----------------------|
| **Pipeline** | None (Direct) | Custom Pipecat | Standard Pipecat |
| **src Integration** | Direct usage | Frame adapters | Not used |
| **Audio Processing** | Byte-level | Frame-based | Frame-based |
| **Complexity** | Simple | Complex | Medium |
| **Port** | 7862 | 7861 | 7860 |
| **Performance** | High | Medium | Medium |

## 🐛 **Troubleshooting**

### **Common Issues**

1. **WebRTC Connection Issues**
   ```bash
   # Check browser console for WebRTC errors
   # Ensure microphone permissions are granted
   ```

2. **Audio Frame Processing**
   ```bash
   # Enable verbose logging
   python direct_server.py --verbose
   ```

3. **src Component Errors**
   ```bash
   # Ensure src directory is accessible
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
   
   # Check Ollama is running
   curl http://localhost:11434/api/tags
   ```

4. **TTS Issues**
   ```bash
   # Check Coqui TTS setup
   # Ensure audio output device is available
   ```

## 🔧 **Development**

### **Adding Features**
To modify the bot behavior:
1. Edit `DirectWebRTCBot` class in `direct_webrtc_bot.py`
2. Modify audio processing logic
3. Update WebRTC message handling

### **Debugging Audio**
```python
# Add debug logging in _process_audio_frame()
logger.debug(f"Audio frame: {len(audio_data)} bytes")
logger.debug(f"VAD result: {is_speech}")
```

### **Testing Components**
Test individual components:
```python
# Test VAD
vad = VADProcessor()
# Test STT  
stt = WhisperSTTAgent()
# Test LLM
llm = OllamaAgent()
# Test TTS
tts = CoquiTTSAgent()
```

## 📝 **Notes**

- **No Pipecat Pipeline**: This approach completely bypasses Pipecat's frame processing
- **Direct Audio Processing**: Uses the same byte-level processing as `example_parallel_sva.py`
- **WebRTC Frontend**: Maintains the professional web interface
- **Threading**: Uses background threads for CPU-intensive tasks
- **Queue Communication**: Similar to multiprocess queues in the example
- **TTS Integration**: Includes Coqui TTS for complete voice interaction

## 🎉 **Benefits**

1. **Simple Integration**: Uses your `src` components as-is
2. **No Pipeline Overhead**: Direct audio processing
3. **Familiar Logic**: Same as `example_parallel_sva.py`
4. **Web Interface**: Professional WebRTC frontend
5. **High Performance**: Minimal processing overhead
6. **Easy Debugging**: Clear separation of concerns 