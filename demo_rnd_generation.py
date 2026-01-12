#!/usr/bin/env python3
"""
Demo script for RND1 generation.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

from transformers import AutoTokenizer

# Add RND1 module to path for local testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def demo_completion(
    model_path: str,
    checkpoint_path: str = None,
    device: str = "cuda:0",
    use_bfloat16: bool = True,
    show_visualization: bool = True,
    num_steps: int = 64,
    max_new_tokens: int = 256,
    custom_prompt: str = None,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    mask_token_id: int = 151669,
    seed: int = None,
    moe_backend: str = "hf",
    mode: str = "task",
    add_eos_at_end: bool = False,
    eb_gamma: float = None,
):
    """
    Demonstrate text completion using RND1.

    Args:
        model_path: Path to base model or HuggingFace model ID
        checkpoint_path: Path to custom checkpoint (if any)
        device: Device to run on (e.g., cuda:0, cpu)
        use_bfloat16: Whether to use bfloat16 precision
        show_visualization: Whether to show live visualization (requires rich)
        num_steps: Number of diffusion steps
        max_new_tokens: Maximum number of tokens to generate
        custom_prompt: Custom prompt to use instead of default examples
        temperature: Temperature for sampling (0.0 = greedy)
        top_k: Top-k filtering for sampling (None = disabled)
        top_p: Top-p (nucleus) filtering for sampling (None = disabled)
        mask_token_id: Token ID for mask token
        seed: Random seed for reproducibility
        moe_backend: MoE backend to use ('hf', 'vllm', 'sglang', 'flashinfer')
        mode: Generation mode ('task' for Q&A format, 'completion' for continuation)
        add_eos_at_end: Whether to add EOS token at the end of the sequence
    """
    # if seed is not None:
    if seed is None:
        # generate a random seed
        seed = random.randint(0, 1000000)
        print(f"Seed not provided, using random seed: {seed}")
    set_seed(seed)

    from rnd.configuration_rnd import RND1Config
    from rnd.modeling_rnd import RND1LM

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"Using dtype: {dtype}")

    if moe_backend == "hf":
        print(
            "\n⚠️  Note: HuggingFace backend is slower. "
            "Consider using --moe_backend vllm, sglang or flashinfer for better performance.\n"
        )

    # Load from checkpoint if provided, otherwise from model_path
    load_path = checkpoint_path if checkpoint_path else model_path

    print(f"Loading model from {load_path}...")

    # Load config and set RND1-specific settings
    cfg = RND1Config.from_pretrained(load_path)
    cfg.model_type = "rnd1"
    cfg.attn_implementation = "sdpa"
    cfg.moe_backend = moe_backend

    # Load model with RND1LM
    model = RND1LM.from_pretrained(
        load_path,
        config=cfg,
        dtype=dtype,
        device_map="auto" if device == "cuda:0" else device,
        trust_remote_code=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    print("Model loaded")
    model = model.eval()

    if custom_prompt:
        prompts = [custom_prompt]
    else:
        # Default prompts based on mode
        if mode == "task":
            prompts = [
                "Write a Python function that finds the longest common subsequence of two strings."
                "Include comments explaining the algorithm."
            ]
        else:
            prompts = ["The key to understanding quantum computing lies in"]

    greedy = temperature == 0.0

    for i, user_prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"Mode: {mode.upper()}")
        print(f"Prompt {i + 1}: {user_prompt[:100]}...")
        print(f"{'=' * 60}\n")

        if mode == "task":
            # Task mode: Add "Question: " prefix if not already present
            if not user_prompt.strip().startswith("Question:"):
                prompt = f"Question: {user_prompt}\nAnswer:"
            else:
                prompt = user_prompt
        else:
            # Completion mode: Use prompt as-is for continuation
            prompt = user_prompt

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device if device != "auto" else "cuda")

        print("Generation parameters:")
        print(f"  Prompt length: {input_ids.shape[1]} tokens")
        print(f"  Max new tokens: {max_new_tokens}")
        print(f"  Total sequence: {input_ids.shape[1] + max_new_tokens} tokens")
        print(f"  Diffusion steps: {num_steps}")
        print(f"  Temperature: {temperature}")
        print(f"  Greedy: {greedy}")
        if top_k:
            print(f"  Top-k: {top_k}")
        if top_p:
            print(f"  Top-p: {top_p}")
        print()

        # Create explicit generation config that takes priority over model defaults
        from rnd.generation_config import RND1GenerationConfig

        gen_config = RND1GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_diffusion_steps=num_steps,
            mask_token_id=mask_token_id,
            temperature=temperature if not greedy else 0.0,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 151645,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            add_eos_at_end=add_eos_at_end,
            eb_gamma=eb_gamma,
        )

        with torch.no_grad():
            if show_visualization and hasattr(model, "generate_with_visualization"):
                # Use method with visualization support (requires tokenizer)
                output = model.generate_with_visualization(
                    tokenizer=tokenizer,
                    inputs=input_ids,
                    generation_config=gen_config,
                )
            else:
                # Use standard generate method with explicit config
                output = model.generate(
                    inputs=input_ids,
                    generation_config=gen_config,
                )

        generated_tokens = output[0][len(input_ids[0]) :]
        generation = tokenizer.decode(generated_tokens.tolist(), skip_special_tokens=True)

        if not show_visualization:  # by default the viz shows final response too
            print("\nGenerated response:")
            print(generation)

        print(f"\n(Generation completed in {num_steps} diffusion steps)")


def main():
    parser = argparse.ArgumentParser(
        description="RND1 diffusion model demo with live visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_path", type=str, default="radicalnumerics/RND1-Base-0910", help="Path to model or HuggingFace model ID"
    )
    model_group.add_argument("--checkpoint", type=str, default=None, help="Path to custom checkpoint file or directory")
    model_group.add_argument("--device", type=str, default="cuda:0", help="Device to run on (e.g., cuda:0, cpu)")
    model_group.add_argument("--fp32", action="store_true", help="Use FP32 precision instead of BF16")

    # Generation configuration
    gen_group = parser.add_argument_group("Generation Settings")
    gen_group.add_argument("--num_steps", type=int, default=256, help="Number of diffusion steps")
    gen_group.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    gen_group.add_argument("--prompt", type=str, default=None, help="Custom prompt to use for generation")
    gen_group.add_argument(
        "--mode",
        type=str,
        default="task",
        choices=["task", "completion"],
        help="Generation mode: 'task' (Q&A format for instructions) or 'completion' (text continuation)",
    )
    gen_group.add_argument("--mask_token_id", type=int, default=151669, help="Token ID for mask token")

    # Sampling configuration
    sampling_group = parser.add_argument_group("Sampling Parameters")
    sampling_group.add_argument(
        "--temperature", type=float, default=0.01, help="Temperature for sampling (0.0 = greedy/deterministic)"
    )
    sampling_group.add_argument(
        "--top_k", type=int, default=None, help="Top-k filtering: keep only k most likely tokens"
    )
    sampling_group.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) filtering: keep tokens with cumulative probability <= p",
    )
    sampling_group.add_argument(
        "--eb_gamma",
        type=float,
        default=None,
        help="EB-Sampler gamma parameter for entropy-based position selection. "
             "If set, enables EB-Sampler instead of default entropy-based selection. "
             "Lower values select fewer positions per step."
    )

    # Visualization
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument(
        "--no_viz", action="store_true", help="Disable live visualization during generation (requires rich library)"
    )

    # Other settings
    other_group = parser.add_argument_group("Other Settings")
    other_group.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")

    moe_backend_group = parser.add_argument_group("MoE Backend")
    moe_backend_group.add_argument(
        "--moe_backend",
        type=str,
        default="hf",
        choices=["hf", "vllm", "sglang", "flashinfer"],
        help="MoE backend to use for sparse mixture of experts layers",
    )
    add_eos_at_end_group = parser.add_argument_group("EOS Token")
    add_eos_at_end_group.add_argument(
        "--add_eos_at_end",
        action="store_true",
        help="Add End of Sequence (EOS) token at the end of the sequence. "
        "This can be useful to force the model to generate a complete sentence.",
    )

    args = parser.parse_args()

    if args.temperature < 0:
        parser.error("Temperature must be non-negative")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("Top-k must be positive")
    if args.top_p is not None and (args.top_p <= 0 or args.top_p > 1):
        parser.error("Top-p must be between 0 and 1")

    print("\n" + "=" * 60)
    print("RND1 Diffusion Language Model Demo")
    print("=" * 60)
    print("Configuration:")
    print(f"  Model: {args.model_path}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print(f"  Precision: {'FP32' if args.fp32 else 'BF16'}")
    print(
        f"  Mode: {args.mode.upper()} ({'Q&A format for instructions' if args.mode == 'task' else 'Text continuation'})"
    )
    print(f"  Random seed: {args.seed}")
    print(f"  Diffusion steps: {args.num_steps}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("  Algorithm: Entropy-based selection")
    print(f"  Temperature: {args.temperature}")
    if args.top_k:
        print(f"  Top-k: {args.top_k}")
    if args.top_p:
        print(f"  Top-p: {args.top_p}")
    if args.eb_gamma:
        print(f"  EB-Sampler gamma: {args.eb_gamma}")
    print(f"  MoE Backend: {args.moe_backend}")
    print(f"  Visualization: {'Enabled' if not args.no_viz else 'Disabled'}")
    print("=" * 60 + "\n")

    demo_completion(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_bfloat16=not args.fp32,
        show_visualization=not args.no_viz,
        num_steps=args.num_steps,
        max_new_tokens=args.max_new_tokens,
        custom_prompt=args.prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        mask_token_id=args.mask_token_id,
        seed=args.seed,
        moe_backend=args.moe_backend,
        mode=args.mode,
        add_eos_at_end=args.add_eos_at_end,
        eb_gamma=args.eb_gamma,
    )


if __name__ == "__main__":
    main()
