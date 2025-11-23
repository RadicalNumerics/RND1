<h1>
<p align="center">
RND1: Scaling Diffusion Language Models
</p>
</h1>

![???](https://github.com/user-attachments/assets/c2c54f94-a7f5-4b76-987d-f15de4efaef6)


This repository contains an inference harness for Radical Numerics Diffusion 1 (RND1), an experimental diffusion language model. RND1-Base-0910 is a 30B‑parameter sparse Mixture‑of‑Experts model with 3B active parameters per token, converted from an autoregressive base (Qwen3-30B-A3B) via continual pretraining on 500B tokens.

We release RND1 models to catalyze further research on inference and post-training of DLMs.

For more details, see:

**Blog:** https://www.radicalnumerics.ai/blog/rnd1

**Report:** https://www.radicalnumerics.ai/assets/rnd1_report.pdf

**🤗:** https://huggingface.co/radicalnumerics/RND1-Base-0910

**Models:**
 * **RND1-Base-0910**: first base model in the RND1 family. It has not been post-trained for specific usage.


## Installation

```bash
# tested with Python 3.12
pip install torch transformers accelerate numpy rich
```

```bash
# backends enable faster inference through optimized MoE kernels:
pip install flashinfer-python
pip install sglang[all]
pip install vllm
```

## Quick Start



```bash
# Task mode (default) - for instructions, questions, or requests
python demo_rnd_generation.py --prompt "Write a Python function that finds the longest common subsequence of two strings. Include comments explaining the algorithm." --moe_backend hf

# Completion mode - for text continuation
python demo_rnd_generation.py --mode completion --prompt "The key to understanding quantum computing lies in" --moe_backend hf

# Sampling parameters
python demo_rnd_generation.py --top_k 50 --temperature 0.7 --prompt "Explain how neural networks learn in simple terms" --moe_backend hf
```


> [!WARNING]
> Selecting a non-Huggingface MoE backend is highly encouraged for faster generation. Note however that non-HF backends currently support a single GPU only, so you need to set e.g. `export CUDA_VISIBLE_DEVICES=0` before running the script. If you use `flashinfer-python`, JIT compilation the first time the code is run may take a while unless `flashinfer-jit-cache` is installed.

### Demo Parameters

- `--mode`: Generation mode - 'task' or 'completion' (default: task)
  - `task`: For instructions, questions, or requests (adds "Question:" prefix)
  - `completion`: For text continuation (no prefix added)
- `--max_new_tokens`: Number of new tokens to generate (default: 256)
- `--num_steps`: Diffusion denoising steps (default: 256)
- `--temperature`: Sampling temperature, 0.0 for greedy (default: 0.01)
- `--top_k`: Top-k filtering - keeps only k most likely tokens (works with greedy and sampling)
- `--top_p`: Nucleus filtering - keeps tokens with cumulative probability ≤ p (works with greedy and sampling)
- `--moe_backend`: Choose backend: hf, vllm, sglang, flashinfer (default: hf)
- `--no_viz`: Disable visualization
- `--add_eos_at_end`: Add End of Sequence (EOS) token at the end of the sequence; useful to force the model to come to a coherent end (default: False)

## Python API

```python
from transformers import AutoTokenizer
from rnd import RND1LM

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("radicalnumerics/RND1-Base-0910", trust_remote_code=True)

# Load model
model = RND1LM.from_pretrained(
    "radicalnumerics/RND1-Base-0910",
    dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True,
    moe_backend="hf", # hf (default), sglang, vllm, flashinfer
)

# Generate - Task mode (for instructions and questions)
prompt = "Write a Python function that finds the longest common subsequence of two strings. Include comments explaining the algorithm."
inputs = tokenizer(f"Question: {prompt}\nAnswer:", return_tensors="pt")
input_ids = inputs.input_ids.to(model.device)

# Generate
output = model.generate(
    inputs=input_ids,
    max_new_tokens=256,
    num_diffusion_steps=256,
    temperature=0.01,
)

# Decode only the generated part
text = tokenizer.decode(output[0], skip_special_tokens=True)
print(text)
```

## Project Structure

```
RND_dev/
├── README.md                    # This file
├── demo_rnd_generation.py       # Demo script with command-line interface
└── rnd/                         # Core RND1 package
    ├── __init__.py              # Package exports
    ├── configuration_rnd.py     # RND1 model configuration
    ├── modeling_rnd.py          # Core model implementation
    ├── generation_config.py     # Generation configuration
    ├── generation_utils.py      # Generation mixin and utilities
    ├── sampling.py              # Diffusion sampling algorithm
    └── terminal_visualizer.py   # Live visualization (optional)
```


---

<p align="center">
  <img width=350 alt="Radical Numerics Logo" src="https://raw.githubusercontent.com/RadicalNumerics/assets/refs/heads/main/svg/rn-logo-desktop-vector-animated.svg" />
</p>
