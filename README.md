# Poor Jarvis

A lightweight, open-source voice assistant built with Python that provides basic voice interaction capabilities without requiring expensive hardware or cloud services. This project aims to make voice assistant technology accessible to everyone.

## Features

- Voice command recognition using local processing
- Text-to-speech capabilities
- Basic command execution
- Customizable wake word detection
- Offline functionality (no internet required for core features)

## Prerequisites

- Python 3.8 or higher
- A working microphone
- Speakers or headphones
- Basic Python development environment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Apoorva87/poor_jarvis.git
cd poor_jarvis
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project includes several example scripts that demonstrate different functionalities:

1. Basic Voice Assistant:
```bash
python example_voice_assistant.py
```

2. Smart Voice Assistant (with additional features):
```bash
python example_smart_voice_assistant.py
```

3. Text-to-Speech Example:
```bash
python example_tts_tacho.py
```

4. Whisper Integration Example:
```bash
python example_whisper.py
```

## Project Structure

- `example_voice_assistant.py`: Basic voice assistant implementation
- `example_smart_voice_assistant.py`: Enhanced voice assistant with additional features
- `example_tts_tacho.py`: Text-to-speech demonstration
- `example_whisper.py`: Integration with OpenAI's Whisper for improved speech recognition
- `remove_emoji.py`: Utility for text processing
- `requirements.txt`: Project dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the MIT License.

## Acknowledgments
- webrtc implementation - git clone https://github.com/pipecat-ai/small-webrtc-prebuilt.git
- Built with Python's speech recognition and text-to-speech libraries
- Inspired by the need for accessible voice assistant technology
