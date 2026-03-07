# ComfyUI-Qwen-TTS

English | [中文版](README_CN.md)

![Nodes Screenshot](example/example.png)

ComfyUI custom nodes for speech synthesis, voice cloning, and voice design, based on the open-source **Qwen3-TTS** project by the Alibaba Qwen team.

## 📋 Changelog

- **2026-03-08**: Feature Update: Refactored **Faster RoleBank** to accept list inputs (`INPUT_IS_LIST`). Added **RoleBank Collector** (8 fixed slots → 3 parallel lists), **Role Accumulator** (stateful, for-loop safe), and **Append Any To List** utility node (`Qwen3-TTS/Utils`).
- **2026-02-04**: Feature Update: Added Global Pause Control (`QwenTTSConfigNode`) and `extra_model_paths.yaml` support ([update.md](doc/update.md))
- **2026-01-29**: Feature Update: Support for loading custom fine-tuned models & speakers ([update.md](doc/update.md))
  - *Note: Fine-tuning is currently experimental; zero-shot cloning is recommended for best results.*
- **2026-01-27**: UI Optimization: Sleek LoadSpeaker UI; fixed PyTorch 2.6+ compatibility ([update.md](doc/update.md))
- **2026-01-26**: Functional Update: New voice persistence system (SaveVoice / LoadSpeaker) ([update.md](doc/update.md))
- **2026-01-24**: Added attention mechanism selection & model memory management features ([update.md](doc/update.md))
- **2026-01-24**: Added generation parameters (top_p, top_k, temperature, repetition_penalty) to all TTS nodes ([update.md](doc/update.md))
- **2026-01-23**: Dependency compatibility & Mac (MPS) support, New nodes: VoiceClonePromptNode, DialogueInferenceNode ([update.md](doc/update.md))

## Online Workflows

- **Qwen3-TTS Multi-Role Multi-Round Dialogue Generation Workflow**:
  - [workflow](https://www.runninghub.ai/post/2014703508829769729/?inviteCode=rh-v1041)
- **Qwen3-TTS 3-in-1 (Clone, Design, Custom) Workflow**:
  - [workflow](https://www.runninghub.ai/post/2014962110224142337/?inviteCode=rh-v1041)

## Key Features

- 🎵 **Speech Synthesis**: High-quality text-to-speech conversion.
- 🎭 **Voice Cloning**: Zero-shot voice cloning from short reference audio.
- 🎨 **Voice Design**: Create custom voice characteristics based on natural language descriptions.
- 🚀 **Efficient Inference**: Supports both 12Hz and 25Hz speech tokenizer architectures.
- 🎯 **Multilingual**: Native support for 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian).
- ⚡ **Integrated Loading**: No separate loader nodes required; model loading is managed on-demand with global caching.
- ⏱️ **Ultra-Low Latency**: Supports high-fidelity speech reconstruction with low-latency streaming.
- 🧠 **Attention Mechanism Selection**: Choose from multiple attention implementations (sage_attn, flash_attn, sdpa, eager) with auto-detection and graceful fallback.
- 💾 **Memory Management**: Optional model unloading after generation to free GPU memory for users with limited VRAM.
- ⚡ **Faster Nodes** (CUDA Graphs): A parallel set of nodes backed by `faster-qwen3-tts` for significantly faster inference via CUDA graph capture.

## Nodes List

### 1. Qwen3-TTS Voice Design (`VoiceDesignNode`)
Generate unique voices based on text descriptions.
- **Inputs**:
  - `text`: Target text to synthesize.
  - `instruct`: Description of the voice (e.g., "A gentle female voice with a high pitch").
  - `model_choice`: Currently locked to **1.7B** for VoiceDesign features.
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.
- **Capabilities**: Best for creating "imaginary" voices or specific character archetypes.

### 2. Qwen3-TTS Voice Clone (`VoiceCloneNode`)
Clone a voice from a reference audio clip.
- **Inputs**:
  - `ref_audio`: A short (5-15s) audio clip to clone.
  - `ref_text`: Text spoken in the `ref_audio` (helps improve quality).
  - `target_text`: The new text you want the cloned voice to say.
  - `model_choice`: Choose between **0.6B** (fast) or **1.7B** (high quality).
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.

### 3. Qwen3-TTS Custom Voice (`CustomVoiceNode`)
Standard TTS using preset speakers.
- **Inputs**:
  - `text`: Target text.
  - `speaker`: Selection from preset voices (Aiden, Eric, Serena, etc.).
  - `instruct`: Optional style instructions.
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.

### 4. Qwen3-TTS Role Bank (`RoleBankNode`) [New]
Collect and manage multiple voice prompts for dialogue generation.
- **Inputs**:
  - Up to 8 roles, each with:
    - `role_name_N`: Name of the role (e.g., "Alice", "Bob", "Narrator")
    - `prompt_N`: Voice clone prompt from `VoiceClonePromptNode`
- **Capabilities**: Create named voice registry for use in `DialogueInferenceNode`. Supports up to 8 different voices per bank.

### 5. Qwen3-TTS Voice Clone Prompt (`VoiceClonePromptNode`) [New]
Extract and reuse voice features from reference audio.
- **Inputs**:
  - `ref_audio`: A short (5-15s) audio clip to extract features from.
  - `ref_text`: Text spoken in the `ref_audio` (highly recommended for better quality).
  - `model_choice`: Choose between **0.6B** (fast) or **1.7B** (high quality).
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.
- **Capabilities**: Extract a "prompt item" once and use it multiple times across different `VoiceCloneNode` instances for faster and more consistent generation.

### 6. Qwen3-TTS Multi-role Dialogue (`DialogueInferenceNode`) [New]
Synthesize complex dialogues with multiple speakers.
- **Inputs**:
  - `script`: Dialogue script in format "RoleName: Text".
  - `role_bank`: Role bank from `RoleBankNode` containing voice prompts.
  - `model_choice`: Choose between **0.6B** (fast) or **1.7B** (high quality).
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.
  - `pause_seconds`: Silence duration between sentences.
  - `merge_outputs`: Merge all dialogue segments into a single long audio.
  - `batch_size`: Number of lines to process in parallel (larger = faster but more VRAM).
- **Capabilities**: Handles multi-role speech synthesis in a single node, ideal for audiobook narration or roleplay scenarios.

### 7. Qwen3-TTS Load Speaker (`LoadSpeakerNode`) [New]
Load saved voice features and metadata with zero configuration.
- **Capabilities**: Enables a "Select & Play" experience by auto-loading pre-computed features and metadata.

### 8. Qwen3-TTS Save Voice (`SaveVoiceNode`) [New]
Persist extracted voice features and metadata to disk for future use.
- **Capabilities**: Build a permanent voice library for reuse via `LoadSpeakerNode`.

### 9. Qwen3-TTS Config (`QwenTTSConfigNode`) [New]
Define global pause durations for punctuation to control speech rhythm.
- **Inputs**:
  - `pause_linebreak`: Silence after linebreaks.
  - `period_pause`: Silence after periods (.).
  - `comma_pause`: Silence after commas (,).
  - `question_pause`: Silence after question marks (?).
  - `hyphen_pause`: Silence after hyphens (-).
- **Usage**: Connect output to the `config` input of other TTS nodes.

---

## ⚡ Faster Nodes (CUDA Graphs)

A parallel set of nodes under the `Qwen3-TTS/Faster` category, backed by the [`faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) library. CUDA Graphs are captured on first run and reused for all subsequent calls, giving a significant latency reduction.

### Installation

```bash
pip install faster-qwen3-tts
```

> All standard model weights are shared — no separate download required.

### Faster Nodes List

#### 10. ⚡ Faster Voice Clone (`FasterQwen3TTSVoiceCloneNode`)
Same purpose as `VoiceCloneNode` but backed by CUDA Graphs.
- **Inputs**: `target_text`, `ref_audio` (AUDIO), `model_choice`, `language`, generation params, `config` (optional)
- **Key extras**: `xvec_only` (x-vector fast mode), `non_streaming_mode`, `append_silence`

#### 11. ⚡ Faster Custom Voice (`FasterQwen3TTSCustomVoiceNode`)
Same purpose as `CustomVoiceNode` but backed by CUDA Graphs.
- **Inputs**: `text`, `speaker`, `language`, `instruct`, generation params, `config` (optional)

#### 12. ⚡ Faster Voice Design (`FasterQwen3TTSVoiceDesignNode`)
Same purpose as `VoiceDesignNode` but backed by CUDA Graphs. Always uses the **1.7B-VoiceDesign** model.
- **Inputs**: `text`, `instruct`, `language`, generation params, `config` (optional)

#### 13. ⚡ Faster Role Bank Collector (`FasterRoleBankCollectorNode`)
Collects up to 8 individual (role_name, audio, ref_text) slots and outputs three parallel **lists** consumed by `FasterRoleBankNode`.
- **Inputs**: Up to 8 groups — `role_name_N`, `audio_N` (AUDIO), `ref_text_N` (optional)
- **Outputs**: `role_name` (STRING list), `audio` (AUDIO list), `ref_text` (STRING list)
- **Usage**: Wire all three outputs to the matching inputs of `FasterRoleBankNode`.

#### 14. ⚡ Faster Role Bank (`FasterRoleBankNode`)
Assembles a `FASTER_ROLE_BANK` dict from three parallel **lists** (uses `INPUT_IS_LIST = True`).
- **Inputs**: `role_name` (STRING list), `audio` (AUDIO list), `ref_text` (STRING list, optional)
- **Output**: `FASTER_ROLE_BANK`
- **Typical source**: `FasterRoleBankCollectorNode` (fixed roles) or `FasterRoleAccumulatorNode` bank_out after a for-loop.

#### 15. ⚡ Faster Role Accumulator (`FasterRoleAccumulatorNode`)
Stateful role bank builder designed to run **inside a for-loop**. Appends one `(role_name, audio, ref_text)` triplet per iteration; outputs a growing `FASTER_ROLE_BANK`.
- **Inputs**: `role_name`, `audio`, `accumulator_id` (unique string key), `reset` (True on first iteration), `ref_text` (optional)
- **Outputs**: `bank_out` (FASTER_ROLE_BANK), `count` (INT)
- **Usage**: Wire `bank_out` through the loop's `loopEnd` node; connect the final bank to `FasterDialogueInferenceNode` **outside** the loop.

#### 16. ⚡ Faster Dialogue Inference (`FasterQwen3TTSDialogueInferenceNode`)
Same purpose as `DialogueInferenceNode` but uses CUDA Graphs for per-line inference.
- **Inputs**: `script`, `faster_role_bank`, `model_choice`, `language`, pause controls, generation params
- **Note**: Processes lines sequentially (no `batch_size`) because the faster library does not support batched `VOICE_CLONE_PROMPT` objects.

### Config (Pause Control) Support

All Faster nodes accept an optional **`config`** input from `QwenTTSConfigNode`. When connected, text is automatically split into segments at punctuation marks and silence is inserted between segments — identical behaviour to the standard nodes.

### Workflow

```
Standard:  LoadAudio → VoiceClonePromptNode → RoleBankNode → DialogueInferenceNode
Faster (fixed roles):   LoadAudio → RoleBankCollectorNode → FasterRoleBankNode → FasterDialogueInferenceNode
Faster (dynamic/loop):  [For Loop] → RoleAccumulatorNode → [loopEnd] → FasterDialogueInferenceNode
```

---

## 🛠️ Utility Nodes (`Qwen3-TTS/Utils`)

### 📋 Append Any To List (`AppendAnyToListNode`)
Appends a single item of **any** ComfyUI type to a list, or starts a new list.
- **Inputs**: `item` (\*, required), `list_in` (\*, optional — must come from a node with `OUTPUT_IS_LIST`)
- **Output**: `list_out` (list, `OUTPUT_IS_LIST = True`)
- **Usage**: Chain multiple nodes to build lists incrementally in static graphs. Not suitable for for-loops (use `FasterRoleAccumulatorNode` instead).

```
[AudioA] → item → [AppendAnyToList] ──list_out──► list_in → [AppendAnyToList] → list_out → [downstream]
                                                              ↑
                                                    [AudioB] → item
```

---

## Attention Mechanisms

All nodes support multiple attention implementations with automatic detection and graceful fallback:

| Mechanism | Description | Speed | Installation |
|-----------|-------------|-------|--------------|
| **sage_attn** | SAGE attention implementation | ⚡⚡⚡ Fastest | `pip install sage_attn` |
| **flash_attn** | Flash Attention 2 | ⚡⚡ Fast | `pip install flash_attn` |
| **sdpa** | Scaled Dot Product Attention (PyTorch built-in) | ⚡ Medium | Built-in (no installation) |
| **eager** | Standard attention (fallback) | 🐢 Slowest | Built-in (no installation) |
| **auto** | Automatically selects best available option | Varies | N/A |

### Auto-Detection Priority

When `attention: "auto"` is selected, the system checks in this order:
1. **sage_attn** → If installed, use SAGE attention (fastest)
2. **flash_attn** → If installed, use Flash Attention 2
3. **sdpa** → Always available (PyTorch built-in)
4. **eager** → Always available (fallback, slowest)

The selected mechanism is logged to the console for transparency.

### Graceful Fallback

If you select an attention mechanism that's not available:
- Falls back to `sdpa` (if available)
- Falls back to `eager` (as last resort)
- Logs the fallback decision with a warning message

### Model Caching

- Models are cached with attention-specific keys
- Changing attention mechanism automatically clears cache and reloads model
- Same model with different attention mechanisms coexists in cache

## Memory Management

### Model Unloading After Generation

The `unload_model_after_generate` toggle is available on all nodes:
- **Enabled**: Clears model cache, GPU memory, and runs garbage collection after generation
- **Disabled**: Model remains in cache for faster subsequent generations (default)

**When to use:**
- ✅ Enable if you have limited VRAM (< 8GB)
- ✅ Enable if you need to run multiple different models sequentially
- ✅ Enable if you're done with generation and want to free memory
- ❌ Disable if you're generating multiple clips with the same model (faster)

**Console Output:**
```
🗑️ [Qwen3-TTS] Unloading 1 cached model(s)...
✅ [Qwen3-TTS] Model cache and GPU memory cleared
```



## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchaudio transformers librosa accelerate
```

### Model Directory Structure

ComfyUI-Qwen-TTS automatically searches for models in the following priority:

```text
ComfyUI/
├── models/
│   └── qwen-tts/
│       ├── Qwen/Qwen3-TTS-12Hz-1.7B-Base/
│       ├── Qwen/Qwen3-TTS-12Hz-0.6B-Base/
│       ├── Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/
│       ├── Qwen/Qwen3-TTS-Tokenizer-12Hz/
│       └── voices/ (Saved presets .wav/.qvp)
```

**Note**: You can also use `extra_model_paths.yaml` to define a custom model path:
```yaml
qwen-tts: D:\MyModels\Qwen
```

## Tips for Best Results

### Audio Quality
- **Cloning**: Use clean, noise-free reference audio (5-15 seconds).
- **Reference Text**: Providing text spoken in reference audio significantly improves quality.
- **Language**: Select the correct language for best pronunciation and prosody.

### Performance & Memory
- **VRAM**: Use `bf16` precision to save significant memory with minimal quality loss.
- **Attention**: Use `attention: "auto"` for automatic selection of fastest available mechanism.
- **Model Unloading**: Enable `unload_model_after_generate` if you have limited VRAM (< 8GB) or need to run multiple different models.
- **Local Models**: Pre-download weights to `models/qwen-tts/` to prioritize local loading and avoid HuggingFace timeouts.

### Attention Mechanisms
- **Best Performance**: Install `sage_attn` or `flash_attn` for 2-3x speedup over sdpa.
- **Compatibility**: Use `sdpa` (default) for maximum compatibility - no installation required.
- **Low VRAM**: Use `eager` with smaller models (0.6B) if other mechanisms cause OOM errors.

### Dialogue Generation
- **Batch Size**: Increase `batch_size` for faster generation (more VRAM usage).
- **Pauses**: Adjust `pause_seconds` to control timing between dialogue segments.
- **Merge**: Enable `merge_outputs` for continuous dialogue; disable for separate clips.

### Faster Nodes
- **First Run**: CUDA Graphs are captured on first inference — expect a warm-up delay. Subsequent calls are significantly faster.
- **Dialogue**: Use `FasterRoleBankNode` (AUDIO inputs) instead of the standard `RoleBankNode` (VOICE_CLONE_PROMPT inputs).
- **Pause config**: Connect `QwenTTSConfigNode` to the `config` input of any Faster node for punctuation-based pause control.

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Official open-source repository by Alibaba Qwen team.

## License

- This project is licensed under the **Apache License 2.0**.
- Model weights are subject to the [Qwen3-TTS License Agreement](https://github.com/QwenLM/Qwen3-TTS#License).

## Author

- **Bilibili**: [Space](https://space.bilibili.com/5594117?spm_id_from=333.1007.0.0)
- **YouTube**: [Channel](https://www.youtube.com/channel/UCx5L-wKf93YNbcP_55vDCeg)
