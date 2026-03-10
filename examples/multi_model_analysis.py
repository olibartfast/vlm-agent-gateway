#!/usr/bin/env python3
"""
Multi-Model Analysis Example

Demonstrates using the Mixture-of-Agents (MoA) workflow to get consensus
from multiple VLM providers on the same image.

Usage:
    OPENAI_API_KEY=... TOGETHER_API_KEY=... python examples/multi_model_analysis.py image.jpg
"""

import argparse
import json
import sys

from vlm_agent_gateway import run_moa
from vlm_agent_gateway.cli import make_agent


def main():
    parser = argparse.ArgumentParser(description="Multi-model image analysis with MoA")
    parser.add_argument("image", help="Image file path or URL")
    parser.add_argument("--prompt", "-p", default="Describe this image in detail.")
    args = parser.parse_args()

    # Create proposer agents from different providers
    try:
        proposers = [
            make_agent("gpt-5-mini", "openai", "https://api.openai.com/v1/chat/completions"),
            make_agent(
                "Qwen/Qwen3.5-VL-72B-Instruct",
                "together",
                "https://api.together.xyz/v1/chat/completions",
            ),
        ]
        aggregator = make_agent("gpt-5", "openai", "https://api.openai.com/v1/chat/completions")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Make sure OPENAI_API_KEY and TOGETHER_API_KEY are set.")
        sys.exit(1)

    result = run_moa(
        proposer_agents=proposers,
        aggregator_agent=aggregator,
        prompt=args.prompt,
        image_paths=[args.image],
        detail="low",
        max_tokens=500,
        resize=False,
        target_size=(512, 512),
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
