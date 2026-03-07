# ComfyUI-Qwen-TTS Faster Nodes Implementation
# Wraps andimarafioti/faster-qwen3-tts (CUDA Graph based inference)

import os
import tempfile
import torch
import numpy as np
import soundfile as sf
from typing import Any, Tuple
import folder_paths
from comfy import model_management

# Reuse pause-splitting logic from the standard nodes
from .nodes import split_text_by_pauses

# Try importing the faster library
try:
    from faster_qwen3_tts import FasterQwen3TTS
    HAS_FASTER_QWEN = True
except ImportError:
    FasterQwen3TTS = None
    HAS_FASTER_QWEN = False
    print("\n⚠️ [Faster Qwen3-TTS] 'faster-qwen3-tts' package is NOT installed.")
    print("   Run: pip install faster-qwen3-tts\n")


DEMO_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "French",
    "German", "Spanish", "Portuguese", "Russian", "Italian",
]

LANGUAGE_MAP = {
    "Auto": "auto", "Chinese": "chinese", "English": "english",
    "Japanese": "japanese", "Korean": "korean", "French": "french",
    "German": "german", "Spanish": "spanish", "Portuguese": "portuguese",
    "Russian": "russian", "Italian": "italian",
}

_FASTER_MODEL_CACHE = {}


def _resolve_model_path(model_choice: str, model_type: str, custom_model_path: str = "") -> str:
    """Resolve local model path from ComfyUI model dirs, or fall back to HF id."""
    HF_MODEL_MAP = {
        ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }
    expected_id = HF_MODEL_MAP.get((model_type, model_choice), "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    if custom_model_path and os.path.isdir(custom_model_path):
        print(f"🔧 [Faster Qwen3-TTS] Using custom model path: {custom_model_path}")
        return custom_model_path

    # Search ComfyUI qwen-tts model directories
    base_paths = []
    try:
        qwen_paths = folder_paths.get_folder_paths("qwen-tts") or []
        base_paths.extend(qwen_paths)
        comfy_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
        default_qwen = os.path.join(comfy_root, "models", "qwen-tts")
        if default_qwen not in base_paths and os.path.exists(default_qwen):
            base_paths.append(default_qwen)
    except Exception:
        pass

    for base in base_paths:
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            cand = os.path.join(base, d)
            if os.path.isdir(cand) and model_choice in d and model_type.lower() in d.lower():
                print(f"✅ [Faster Qwen3-TTS] Found local model: {cand}")
                return cand

    print(f"🌐 [Faster Qwen3-TTS] Local not found, using HuggingFace Hub: {expected_id}")
    return expected_id


def load_faster_model(model_choice: str, model_type: str, custom_model_path: str = "") -> Any:
    if not HAS_FASTER_QWEN:
        raise RuntimeError("'faster-qwen3-tts' is not installed. Run: pip install faster-qwen3-tts")

    global _FASTER_MODEL_CACHE
    cache_key = (model_choice, model_type, custom_model_path)

    if cache_key in _FASTER_MODEL_CACHE:
        return _FASTER_MODEL_CACHE[cache_key]

    model_path = _resolve_model_path(model_choice, model_type, custom_model_path)
    print(f"⌛ [Faster Qwen3-TTS] Loading model (CUDA graphs will capture on first use)...")
    model = FasterQwen3TTS.from_pretrained(model_path)
    _FASTER_MODEL_CACHE[cache_key] = model
    return model


def unload_faster_model(cache_key=None):
    """Unload cached faster model(s) and free GPU memory.

    Args:
        cache_key: If provided (tuple of model_choice, model_type, custom_model_path),
                   only that specific model is unloaded.
                   If None, all cached models are unloaded.
    """
    global _FASTER_MODEL_CACHE
    if cache_key is not None:
        if cache_key in _FASTER_MODEL_CACHE:
            print(f"[Faster Qwen3-TTS] Unloading model: {cache_key}")
            del _FASTER_MODEL_CACHE[cache_key]
        else:
            print(f"[Faster Qwen3-TTS] Cache key not found, nothing to unload: {cache_key}")
    else:
        if _FASTER_MODEL_CACHE:
            print(f"[Faster Qwen3-TTS] Unloading {len(_FASTER_MODEL_CACHE)} cached model(s)...")
            _FASTER_MODEL_CACHE.clear()
    model_management.soft_empty_cache()
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def audio_to_tempfile(audio_input) -> str:
    """
    Convert ComfyUI audio dict (waveform tensor + sample_rate) to a temp WAV file path.
    faster-qwen3-tts always expects a file path for ref_audio.
    """
    waveform = audio_input.get("waveform")
    sr = audio_input.get("sample_rate", 24000)

    if isinstance(waveform, torch.Tensor):
        wav = waveform.squeeze().cpu().numpy()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
    elif isinstance(waveform, np.ndarray):
        wav = waveform.squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
    else:
        raise ValueError(f"Unsupported waveform type: {type(waveform)}")

    wav = wav.astype(np.float32)
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tf.name
    tf.close()  # Close before writing — required on Windows (file lock)
    try:
        sf.write(tmp_path, wav, sr)
    except Exception:
        os.unlink(tmp_path)
        raise
    return tmp_path


def waveform_to_comfy(audio_arrays, sr: int) -> dict:
    """Convert list of numpy arrays back to ComfyUI AUDIO format."""
    wav = torch.from_numpy(audio_arrays[0]).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)   # (1, 1, S)
    return {"waveform": wav, "sample_rate": sr}


def _generate_with_config(generate_fn, text: str, config, sr_default: int = 24000, **kwargs) -> dict:
    """
    Run generate_fn (returns (audio_arrays, sr)) over each text segment produced
    by split_text_by_pauses, then concatenate with silence inserted between segments.

    Args:
        generate_fn: callable(**kwargs) -> (audio_arrays, sr)
        text:        full input text
        config:      TTS_CONFIG dict or None
        sr_default:  fallback sample rate before any segment runs
        **kwargs:    forwarded verbatim to generate_fn (except 'text')
    Returns:
        ComfyUI AUDIO dict
    """
    segments = split_text_by_pauses(text, config)
    results = []
    sr = sr_default

    for seg_text, pause_dur in segments:
        if not seg_text.strip():
            if pause_dur > 0 and results:
                silence = torch.zeros(1, 1, int(pause_dur * sr))
                results.append(silence)
            continue

        audio_arrays, sr = generate_fn(text=seg_text, **kwargs)
        if not audio_arrays:
            raise RuntimeError("Faster engine returned empty audio.")

        wav_np = audio_arrays[0]
        wav = torch.from_numpy(wav_np).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)   # (1, 1, S)
        results.append(wav)

        if pause_dur > 0:
            silence = torch.zeros(1, 1, int(pause_dur * sr))
            results.append(silence)

    if not results:
        raise RuntimeError("No audio segments produced.")

    merged = torch.cat(results, dim=-1)
    return {"waveform": merged, "sample_rate": sr}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Faster Voice Clone
# API: generate_voice_clone(text, language, ref_audio, ref_text,
#                           xvec_only, non_streaming_mode, ...)
# ─────────────────────────────────────────────────────────────────────────────
class FasterQwen3TTSVoiceCloneNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_text": ("STRING", {"multiline": True, "default": "Hello, how are you today?"}),
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Transcript of reference audio"}),
                "xvec_only": ("BOOLEAN", {"default": True, "tooltip": "True=fast x-vector only; False=full ICL (slower, more faithful)"}),
                "non_streaming_mode": ("BOOLEAN", {"default": True, "tooltip": "Full-context generation for better prosody"}),
                "append_silence": ("BOOLEAN", {"default": True, "tooltip": "Append 0.5s silence to ref audio to avoid phoneme bleed at start"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "tooltip": "-1 = random"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "min_new_tokens": ("INT", {"default": 2, "min": 1, "max": 256}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
                "custom_model_path": ("STRING", {"default": ""}),
                "config": ("TTS_CONFIG", {"tooltip": "Optional pause config from QwenTTSConfigNode"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS/Faster"
    DESCRIPTION = "⚡ Faster Voice Clone (CUDA Graphs)"

    def generate(self, target_text, ref_audio, model_choice, language,
                 ref_text="", xvec_only=True, non_streaming_mode=True,
                 append_silence=True, seed=-1,
                 max_new_tokens=2048, min_new_tokens=2,
                 temperature=0.9, top_k=50, top_p=1.0,
                 do_sample=True, repetition_penalty=1.05,
                 unload_model_after_generate=False, custom_model_path="",
                 config=None):

        if seed >= 0:
            torch.manual_seed(seed)

        cache_key = (model_choice, "Base", custom_model_path)
        model = load_faster_model(model_choice, "Base", custom_model_path)
        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        # faster-qwen3-tts needs a file path for ref_audio
        tmp_path = audio_to_tempfile(ref_audio)
        try:
            print(f"⚡ [Faster Qwen3-TTS] VoiceClone: '{target_text[:40]}...'")
            shared_kwargs = dict(
                language=mapped_lang,
                ref_audio=tmp_path,
                ref_text=ref_text if ref_text.strip() else "",
                xvec_only=xvec_only,
                non_streaming_mode=non_streaming_mode,
                append_silence=append_silence,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )
            audio_data = _generate_with_config(model.generate_voice_clone, target_text, config, **shared_kwargs)
        finally:
            os.unlink(tmp_path)

        if unload_model_after_generate:
            unload_faster_model(cache_key)

        return (audio_data,)


# ─────────────────────────────────────────────────────────────────────────────
# Node: Faster Custom Voice
# API: generate_custom_voice(text, speaker, language, instruct, ...)
#      NOTE: no non_streaming_mode parameter
# ─────────────────────────────────────────────────────────────────────────────
class FasterQwen3TTSCustomVoiceNode:

    @classmethod
    def INPUT_TYPES(cls):
        speakers = ["aiden", "allison", "ava", "chloe", "cody", "emma",
                    "eric", "jenna", "liam", "michael", "olivia", "serena"]
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, how are you today?"}),
                # "speaker": (speakers,),
                "speaker": ("STRING", {"default": "Serena"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "tooltip": "-1 = random"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "min_new_tokens": ("INT", {"default": 2, "min": 1, "max": 256}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
                "custom_model_path": ("STRING", {"default": ""}),
                "config": ("TTS_CONFIG", {"tooltip": "Optional pause config from QwenTTSConfigNode"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS/Faster"
    DESCRIPTION = "⚡ Faster Custom Voice (Pre-defined Speakers)"

    def generate(self, text, speaker, model_choice, language,
                 instruct="", seed=-1,
                 max_new_tokens=2048, min_new_tokens=2,
                 temperature=0.9, top_k=50, top_p=1.0,
                 do_sample=True, repetition_penalty=1.05,
                 unload_model_after_generate=False, custom_model_path="",
                 config=None):

        if seed >= 0:
            torch.manual_seed(seed)

        cache_key = (model_choice, "CustomVoice", custom_model_path)
        model = load_faster_model(model_choice, "CustomVoice", custom_model_path)
        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        print(f"⚡ [Faster Qwen3-TTS] CustomVoice ({speaker}): '{text[:40]}...'")
        shared_kwargs = dict(
            speaker=speaker,
            language=mapped_lang,
            instruct=instruct if instruct.strip() else None,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        audio_data = _generate_with_config(model.generate_custom_voice, text, config, **shared_kwargs)

        if unload_model_after_generate:
            unload_faster_model(cache_key)

        return (audio_data,)


# ─────────────────────────────────────────────────────────────────────────────
# Node: Faster Voice Design
# API: generate_voice_design(text, instruct, language, ...)
#      NOTE: no non_streaming_mode parameter
# ─────────────────────────────────────────────────────────────────────────────
class FasterQwen3TTSVoiceDesignNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Welcome to the show."}),
                "instruct": ("STRING", {"multiline": True, "default": "Warm, confident narrator with slight British accent"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "tooltip": "-1 = random"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "min_new_tokens": ("INT", {"default": 2, "min": 1, "max": 256}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
                "custom_model_path": ("STRING", {"default": ""}),
                "config": ("TTS_CONFIG", {"tooltip": "Optional pause config from QwenTTSConfigNode"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS/Faster"
    DESCRIPTION = "⚡ Faster Voice Design (Attributes, 1.7B-VoiceDesign model)"

    def generate(self, text, instruct, language,
                 seed=-1,
                 max_new_tokens=2048, min_new_tokens=2,
                 temperature=0.9, top_k=50, top_p=1.0,
                 do_sample=True, repetition_penalty=1.05,
                 unload_model_after_generate=False, custom_model_path="",
                 config=None):

        if seed >= 0:
            torch.manual_seed(seed)

        # VoiceDesign is always 1.7B
        cache_key = ("1.7B", "VoiceDesign", custom_model_path)
        model = load_faster_model("1.7B", "VoiceDesign", custom_model_path)
        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        print(f"⚡ [Faster Qwen3-TTS] VoiceDesign: '{text[:40]}...'")
        shared_kwargs = dict(
            instruct=instruct,
            language=mapped_lang,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        audio_data = _generate_with_config(model.generate_voice_design, text, config, **shared_kwargs)

        if unload_model_after_generate:
            unload_faster_model(cache_key)

        return (audio_data,)


# ─────────────────────────────────────────────────────────────────────────────
# Node: Faster Role Bank
# Stores raw ComfyUI AUDIO per role (faster lib needs file paths, not prompt objs)
# ─────────────────────────────────────────────────────────────────────────────
class FasterRoleBankNode:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {}, "optional": {}}
        for i in range(1, 9):
            inputs["optional"][f"role_name_{i}"] = (
                "STRING", {"default": f"Role{i}", "tooltip": f"Name of role {i}"}
            )
            inputs["optional"][f"audio_{i}"] = (
                "AUDIO", {"tooltip": f"Reference audio for role {i}"}
            )
            inputs["optional"][f"ref_text_{i}"] = (
                "STRING", {"default": "", "tooltip": f"Transcript of reference audio for role {i} (optional)"}
            )
        return inputs

    RETURN_TYPES = ("FASTER_ROLE_BANK",)
    RETURN_NAMES = ("faster_role_bank",)
    FUNCTION = "create_bank"
    CATEGORY = "Qwen3-TTS/Faster"
    DESCRIPTION = "⚡ Faster RoleBank: Map role names to reference AUDIO clips for dialogue inference."

    def create_bank(self, **kwargs):
        """
        Returns a dict: {role_name: {"audio": <ComfyUI AUDIO dict>, "ref_text": str}}
        """
        bank = {}
        for i in range(1, 9):
            name = kwargs.get(f"role_name_{i}", "").strip()
            audio = kwargs.get(f"audio_{i}")
            ref_text = kwargs.get(f"ref_text_{i}", "").strip()
            if name and audio is not None:
                bank[name] = {"audio": audio, "ref_text": ref_text}
        return (bank,)


# ─────────────────────────────────────────────────────────────────────────────
# Node: Faster Dialogue Inference
# ─────────────────────────────────────────────────────────────────────────────
class FasterQwen3TTSDialogueInferenceNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Role1: Hello, how are you?\nRole2: I am fine, thank you.",
                        "placeholder": "Format: RoleName: Text  (one line per utterance)",
                    },
                ),
                "faster_role_bank": ("FASTER_ROLE_BANK",),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
                "pause_linebreak": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence between lines"}),
                "period_pause":    ("FLOAT", {"default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence after periods (.)"}),
                "comma_pause":     ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence after commas (,)"}),
                "question_pause":  ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence after question marks (?)"}),
                "hyphen_pause":    ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence after hyphens (-)"}),
            },
            "optional": {
                "xvec_only": ("BOOLEAN", {"default": True, "tooltip": "True=fast x-vector only; False=full ICL (slower)"}),
                "non_streaming_mode": ("BOOLEAN", {"default": True}),
                "append_silence": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "tooltip": "-1 = random"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "min_new_tokens": ("INT", {"default": 2,    "min": 1, "max": 256}),
                "temperature":        ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.01}),
                "top_k":              ("INT",   {"default": 50,  "min": 0,    "max": 1000}),
                "top_p":              ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "do_sample":          ("BOOLEAN", {"default": True}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
                "custom_model_path":  ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_dialogue"
    CATEGORY = "Qwen3-TTS/Faster"
    DESCRIPTION = "⚡ Faster DialogueInference: Multi-role script → audio (CUDA Graphs)"

    def generate_dialogue(
        self,
        script: str,
        faster_role_bank: dict,
        model_choice: str,
        language: str,
        pause_linebreak: float,
        period_pause: float,
        comma_pause: float,
        question_pause: float,
        hyphen_pause: float,
        xvec_only: bool = True,
        non_streaming_mode: bool = True,
        append_silence: bool = True,
        seed: int = -1,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        unload_model_after_generate: bool = False,
        custom_model_path: str = "",
    ):
        import re

        if not script or not faster_role_bank:
            raise RuntimeError("Script and Faster Role Bank are required.")

        if seed >= 0:
            torch.manual_seed(seed)

        cache_key = (model_choice, "Base", custom_model_path)
        model = load_faster_model(model_choice, "Base", custom_model_path)
        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        # Build inline pause config dict (mirrors QwenTTSConfigNode output)
        inline_config = {
            "period_pause":   period_pause,
            "comma_pause":    comma_pause,
            "question_pause": question_pause,
            "hyphen_pause":   hyphen_pause,
            "pause_linebreak": pause_linebreak,  # stored but not used by split_text_by_pauses
        }

        lines = script.strip().split("\n")
        results = []
        sr = 24000

        # Pre-convert all role audio to temp files to avoid repeated I/O inside the loop
        tmp_files = {}  # role_name -> tmp_path
        try:
            for role_name, role_data in faster_role_bank.items():
                tmp_files[role_name] = audio_to_tempfile(role_data["audio"])

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Support both ASCII ':' and full-width '：'
                pos_en = line.find(":")
                pos_cn = line.find("：")

                if pos_en == -1 and pos_cn == -1:
                    continue

                if pos_en != -1 and (pos_cn == -1 or pos_en < pos_cn):
                    role_name, text = line.split(":", 1)
                else:
                    role_name, text = line.split("：", 1)

                role_name = role_name.strip()
                text = text.strip()

                if role_name not in faster_role_bank:
                    print(f"⚠️ [Faster Dialogue] Role '{role_name}' not in role bank, skipping.")
                    continue

                ref_audio_path = tmp_files[role_name]
                ref_text = faster_role_bank[role_name].get("ref_text", "")

                # Split the line by punctuation pauses (reuse shared helper)
                segments = split_text_by_pauses(text, inline_config)

                for seg_text, seg_pause in segments:
                    if not seg_text.strip():
                        if seg_pause > 0 and results:
                            results.append(torch.zeros(1, 1, int(seg_pause * sr)))
                        continue

                    print(f"⚡ [Faster Dialogue] {role_name}: '{seg_text[:40]}...'")
                    audio_arrays, sr = model.generate_voice_clone(
                        text=seg_text,
                        language=mapped_lang,
                        ref_audio=ref_audio_path,
                        ref_text=ref_text if ref_text else "",
                        xvec_only=xvec_only,
                        non_streaming_mode=non_streaming_mode,
                        append_silence=append_silence,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                    )

                    if not audio_arrays:
                        print(f"⚠️ [Faster Dialogue] Empty audio for segment, skipping.")
                        continue

                    wav = torch.from_numpy(audio_arrays[0]).float()
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0).unsqueeze(0)
                    results.append(wav)

                    if seg_pause > 0:
                        results.append(torch.zeros(1, 1, int(seg_pause * sr)))

                # Inter-line pause
                if pause_linebreak > 0 and results:
                    results.append(torch.zeros(1, 1, int(pause_linebreak * sr)))

        finally:
            # Clean up all temp files even if an error occurs mid-script
            for p in tmp_files.values():
                try:
                    os.unlink(p)
                except OSError:
                    pass

        if not results:
            raise RuntimeError("No dialogue lines were successfully generated.")

        merged = torch.cat(results, dim=-1)

        if unload_model_after_generate:
            unload_faster_model(cache_key)

        return ({"waveform": merged, "sample_rate": sr},)


# ─────────────────────────────────────────────────────────────────────────────
# Exported mappings (consumed by __init__.py)
# ─────────────────────────────────────────────────────────────────────────────
FASTER_NODE_CLASS_MAPPINGS = {
    "FB_FasterQwen3TTSVoiceClone":        FasterQwen3TTSVoiceCloneNode,
    "FB_FasterQwen3TTSCustomVoice":       FasterQwen3TTSCustomVoiceNode,
    "FB_FasterQwen3TTSVoiceDesign":       FasterQwen3TTSVoiceDesignNode,
    "FB_FasterQwen3TTSRoleBank":          FasterRoleBankNode,
    "FB_FasterQwen3TTSDialogueInference": FasterQwen3TTSDialogueInferenceNode,
}

FASTER_NODE_DISPLAY_NAME_MAPPINGS = {
    "FB_FasterQwen3TTSVoiceClone":        "⚡ Faster Qwen3-TTS VoiceClone",
    "FB_FasterQwen3TTSCustomVoice":       "⚡ Faster Qwen3-TTS CustomVoice",
    "FB_FasterQwen3TTSVoiceDesign":       "⚡ Faster Qwen3-TTS VoiceDesign",
    "FB_FasterQwen3TTSRoleBank":          "⚡ Faster Qwen3-TTS RoleBank",
    "FB_FasterQwen3TTSDialogueInference": "⚡ Faster Qwen3-TTS DialogueInference",
}
