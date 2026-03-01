# Qwen3-TTS with MLX-Audio

Text-to-speech using Qwen3-TTS models on macOS. Accepts input from text files or command line.

## Prerequisites

- macOS (Intel or Apple Silicon)
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- [Homebrew](https://brew.sh/) for system dependencies
- ffmpeg, sox (optional, for audio processing)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install ffmpeg sox
```

## Installation

```bash
# Create virtual environment and install
uv venv
uv sync

# Or install in editable mode
uv pip install -e .
```

## Downloading Model Weights

Models are available from [Qwen](https://huggingface.co/Qwen) on HuggingFace. Download the original (non-MLX) models:

```bash
# Install huggingface-hub
pip install huggingface-hub

# Create models directory
mkdir -p models

# Download VoiceDesign model (for voice design via text prompt)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign

# Download CustomVoice model (pre-defined voices: Ryan, Aiden, etc.)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice

# Download Base model (for voice cloning from reference audio)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

## Note on MLX Backend

The MLX community models (e.g., `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`) have compatibility issues with the current mlx-audio library and produce noise. Use the **original Qwen models** with the default transformers backend for reliable results.
```

## Usage

### Voice Design (Text Prompt)

Create a voice using a text description:

```bash
uv run qwen3tts input.txt -o output.wav \
    --voice-design "female, British narrator, calm and professional" \
    -v
```

### Pre-defined Voices

Use built-in voices (Ryan, Aiden, etc.):

```bash
uv run qwen3tts input.txt -o output.wav \
    --voice-name Ryan \
    --voice-style excited \
    -v
```

### Voice Cloning (Base Model)

Clone a voice from a reference audio file (3-10 seconds). You can either provide a transcript or let the model extract the voice embedding automatically:

```bash
# Create voices directory
mkdir -p voices

# Your reference audio (3-10 seconds of speech)
# Save audio as voices/ref.wav

# Option 1: With transcript (better quality)
echo "Transcript of your reference audio here" > voices/ref.txt

uv run qwen3tts input.txt -o output.wav \
    --voice-audio voices/ref.wav \
    --voice-text voices/ref.txt \
    -v

# Option 2: Without transcript (faster, uses voice embedding only)
uv run qwen3tts input.txt -o output.wav \
    --voice-audio voices/ref.wav \
    -v
```

### Command Line Input

```bash
# Read from stdin
echo "Hello world" | uv run qwen3tts - -o hello.wav

# Or use - as input file
cat myfile.txt | uv run qwen3tts - -o output.wav
```

### MP3 Output

```bash
# Requires ffmpeg installed
uv run qwen3tts input.txt -o output.mp3 --format mp3
```

## Project Structure

```
qwen3-tts-mlx/
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── src/
│   └── qwen3tts/
│       ├── __init__.py     # Package init
│       ├── cli.py          # CLI interface
│       └── generate.py     # TTS generation
├── models/                 # Downloaded model weights
├── voices/                 # Reference audio files
└── output/                 # Generated audio files
```

## Available Models

| Model | Use Case |
|-------|----------|
| Qwen3-TTS-12Hz-1.7B-Base | Voice cloning from reference audio |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | Create voice from text description |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | Pre-defined voices (Ryan, Aiden, etc.) |

## Configuration

You can also configure the model path in code:

```python
from qwen3tts.generate import TTSGenerator

generator = TTSGenerator(
    model_path="models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    model_type="design",
    speaker_design="female, American, calm",
    verbose=True,
)

generator.generate("Your text here", "output.wav")
```

## Performance Tips

- Install flash-attn for faster inference: `pip install flash-attn`
- Use the smaller 0.6B models if memory is limited
- For best voice cloning results, use clean reference audio with clear speech