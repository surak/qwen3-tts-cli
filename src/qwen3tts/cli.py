import argparse
import os
import sys
from pathlib import Path

from qwen3tts.generate import TTSGenerator


DEFAULT_MODEL_PATH = "models/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_OUTPUT_DIR = "output"


def get_model_type(args) -> str:
    if args.voice_audio:
        return "base"
    elif args.voice_design:
        return "design"
    elif args.voice_name:
        return "custom"
    else:
        return "base"


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS - Generate speech from text files or CLI input"
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Text file to convert to speech (or - for stdin)",
    )

    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output audio file path (default: output/<input_name>.<format>)",
    )

    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model directory (default: models/Qwen3-TTS-12Hz-1.7B-Base)",
    )

    parser.add_argument(
        "--format",
        choices=["wav", "mp3"],
        default="mp3",
        help="Output audio format (default: mp3)",
    )

    parser.add_argument(
        "--voice-audio",
        metavar="PATH",
        help="Path to reference audio file for voice cloning",
    )

    parser.add_argument(
        "--voice-text",
        metavar="PATH",
        help="Path to transcript of reference audio (required with --voice-audio)",
    )

    parser.add_argument(
        "--voice-design",
        metavar="PROMPT",
        help="Voice design prompt (e.g., 'female, British narrator')",
    )

    parser.add_argument(
        "--voice-name",
        choices=["Ryan", "Aiden"],
        help="Pre-defined voice name (Ryan or Aiden)",
    )

    parser.add_argument(
        "--voice-style",
        default="neutral",
        help="Voice style for CustomVoice model (default: neutral)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--backend",
        choices=["mlx", "transformers"],
        default="transformers",
        help="Backend to use for TTS (default: transformers)",
    )

    args = parser.parse_args()

    if args.voice_design and args.model == DEFAULT_MODEL_PATH:
        args.model = "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    elif args.voice_name and args.model == DEFAULT_MODEL_PATH:
        args.model = "models/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    if args.voice_audio and not os.path.exists(args.voice_audio):
        print(f"Error: Reference audio file not found: {args.voice_audio}", file=sys.stderr)
        sys.exit(1)

    if args.voice_text and not os.path.exists(args.voice_text):
        print(f"Error: Reference text file not found: {args.voice_text}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}", file=sys.stderr)
        print("\nPlease download the model weights first. See README.md for instructions.")
        sys.exit(1)

    if args.input == "-":
        text = sys.stdin.read()
    elif args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(args.input, "r") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = args.output
    elif args.input and args.input != "-":
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        input_name = Path(args.input).stem
        output_path = f"{DEFAULT_OUTPUT_DIR}/{input_name}.{args.format}"
    else:
        output_path = f"{DEFAULT_OUTPUT_DIR}/output.{args.format}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if args.verbose:
        print(f"Model: {args.model}")
        print(f"Input: {len(text)} characters")
        print(f"Output: {output_path}")

    generator = TTSGenerator(
        model_path=args.model,
        model_type=get_model_type(args),
        speaker_audio=args.voice_audio,
        speaker_text=args.voice_text,
        speaker_design=args.voice_design,
        speaker_voice=args.voice_name,
        speaker_instruct=args.voice_style,
        verbose=args.verbose,
        backend=args.backend,
    )

    try:
        generator.generate(text, output_path, args.format)
        print(f"Audio saved to: {output_path}")
    except Exception as e:
        print(f"Error generating audio: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()