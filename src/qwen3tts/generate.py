import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import soundfile as sf


def write_audio(path: str, audio, sample_rate: int):
    if hasattr(audio, 'asnumpy'):
        audio = audio.asnumpy()
    elif hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()

    audio_format = Path(path).suffix.lower()
    
    if audio_format == ".mp3":
        wav_path = path.replace(".mp3", ".wav")
        sf.write(wav_path, audio, sample_rate)
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-q:a", "2", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(wav_path)
    else:
        sf.write(path, audio, sample_rate)


class TTSGenerator:
    def __init__(
        self,
        model_path: str,
        model_type: str = "base",
        speaker_audio: Optional[str] = None,
        speaker_text: Optional[str] = None,
        speaker_design: Optional[str] = None,
        speaker_voice: Optional[str] = None,
        speaker_instruct: Optional[str] = None,
        verbose: bool = True,
        backend: str = "transformers",
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.speaker_audio = speaker_audio
        self.speaker_text = speaker_text
        self.speaker_design = speaker_design
        self.speaker_voice = speaker_voice
        self.speaker_instruct = speaker_instruct
        self.verbose = verbose
        self.backend = backend
        self.model = None

    def _load_model(self):
        if self.model is None:
            if self.backend == "mlx":
                if self.verbose:
                    print(f"Loading MLX model from {self.model_path}...")
                from mlx_audio.tts.utils import load_model
                
                import json
                config_path = Path(self.model_path) / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    if config.get("model_type") == "qwen3_tts" and "talker_config" in config:
                        talker = config["talker_config"]
                        for key in ["hidden_size", "num_hidden_layers", "intermediate_size", 
                                    "num_attention_heads", "rms_norm_eps", "vocab_size",
                                    "num_key_value_heads", "max_position_embeddings", 
                                    "rope_theta", "head_dim", "rope_scaling"]:
                            if key in talker and key not in config:
                                config[key] = talker[key]
                        if "tie_word_embeddings" not in config:
                            config["tie_word_embeddings"] = False
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=2)
                
                self.model = load_model(self.model_path, strict=False)
            else:
                if self.verbose:
                    print(f"Loading transformers model from {self.model_path}...")
                from qwen_tts import Qwen3TTSModel
                import torch
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_path,
                    device_map="cpu",
                    dtype=torch.float32,
                )
            
        return self.model

    def generate(self, text: str, output_path: str, audio_format: str = "wav"):
        model = self._load_model()
        start_time = time.perf_counter()

        if self.backend == "mlx":
            self._generate_mlx(model, text, output_path)
        else:
            self._generate_transformers(text, output_path)

        elapsed = time.perf_counter() - start_time
        char_count = len(text)
        cpm = int(60 * char_count / elapsed) if elapsed > 0 else 0
        if self.verbose:
            print(f"Generated audio in {elapsed:.1f}s ({cpm:,} cpm)")

    def _generate_mlx(self, model, text: str, output_path: str):
        import mlx.core as mx
        import numpy as np
        
        # Determine voice parameter based on mode
        if self.model_type == "design":
            voice = self.speaker_design or "female"
        elif self.model_type == "custom":
            voice = self.speaker_voice or "Ryan"
        else:
            voice = self.speaker_voice or "female"
        
        # Handle reference audio for cloning
        ref_audio = None
        ref_text = None
        if self.model_type == "base" and self.speaker_audio:
            from mlx_audio.audio_io import read as audio_read
            ref_audio, sr = audio_read(self.speaker_audio)
            ref_audio = mx.array(ref_audio)
            
            if self.speaker_text:
                with open(self.speaker_text) as f:
                    ref_text = f.read()
            else:
                if self.verbose:
                    print("Ref_text not found. Transcribing ref_audio...")
                from mlx_audio.stt import load
                stt_model = load("mlx-community/whisper-large-v3-turbo-asr-fp16")
                ref_text = stt_model.generate(ref_audio).text
                if self.verbose:
                    print(f"Ref_text: {ref_text}")
                del stt_model
                mx.clear_cache()
        
        # Generate using MLX
        if self.model_type == "base" and ref_audio is not None:
            results = model.generate(
                text=text,
                lang_code="English",
                temperature=0.6,
                top_p=0.8,
                verbose=self.verbose,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        else:
            results = model.generate(
                text=text,
                voice=voice,
                temperature=0.6,
                top_p=0.8,
                verbose=self.verbose,
            )
        
        # Collect audio from results
        audio_segments = []
        for result in results:
            if hasattr(result, 'audio') and result.audio is not None:
                audio_segments.append(result.audio)
        
        if audio_segments:
            audio = mx.concatenate(audio_segments, axis=0)
            write_audio(output_path, audio, model.sample_rate)

    def _generate_transformers(self, text: str, output_path: str):
        from qwen_tts import Qwen3TTSModel
        import torch
        
        model = Qwen3TTSModel.from_pretrained(
            self.model_path,
            device_map="cpu",
            dtype=torch.float32,
        )
        
        if self.model_type == "base":
            if self.speaker_audio:
                voice_prompt = model.create_voice_clone_prompt(
                    ref_audio=self.speaker_audio,
                    x_vector_only_mode=True,
                )
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="English",
                    voice_clone_prompt=voice_prompt,
                )
            else:
                raise ValueError("Reference audio required for voice cloning")
        elif self.model_type == "design":
            wavs, sr = model.generate_voice_design(
                text=text,
                language="English",
                instruct=self.speaker_design,
            )
        elif self.model_type == "custom":
            wavs, sr = model.generate_custom_voice(
                text=text,
                language="English",
                speaker=self.speaker_voice,
                instruct=self.speaker_instruct,
            )
        
        write_audio(output_path, wavs[0], sr)