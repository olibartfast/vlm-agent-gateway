"""
Command-line interface for VLM Agent Gateway.
"""

import argparse
import json
import logging
import os
import sys
import uuid

from vlm_agent_gateway.config import (
    DEFAULT_ENDPOINT,
    PROVIDER_ENV_MAP,
    resolve_endpoint,
)
from vlm_agent_gateway.models import Agent
from vlm_agent_gateway.tools import BUILTIN_TOOLS
from vlm_agent_gateway.workflows import (
    WORKFLOW_REGISTRY,
    run_conditional,
    run_iterative,
    run_moa,
    run_monitoring,
    run_react,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def make_agent(model: str, provider: str, endpoint: str) -> Agent:
    """Create a single Agent from model/provider/endpoint."""
    env_var = PROVIDER_ENV_MAP.get(provider.lower(), "OPENAI_API_KEY")
    api_key = os.getenv(env_var, "")

    # Allow empty API key for local endpoints
    if not api_key and "localhost" not in endpoint and "127.0.0.1" not in endpoint:
        raise RuntimeError(f"{env_var} not set (required for provider '{provider}')")
    if not api_key:
        api_key = "EMPTY"  # vLLM / SGLang local server convention

    return Agent(model=model, endpoint=endpoint, api_key=api_key, provider=provider)


def build_agents(models: list[str], providers: list[str], endpoints: list[str]) -> list[Agent]:
    """Create list of Agents from parallel model/provider/endpoint lists."""
    n = len(models)
    providers = providers or ["openai"] * n
    endpoints = endpoints or [resolve_endpoint(provider) for provider in providers]

    if len(providers) != n:
        raise ValueError(f"--providers length ({len(providers)}) must match --models length ({n})")
    if len(endpoints) != n:
        raise ValueError(f"--endpoints length ({len(endpoints)}) must match --models length ({n})")

    return [make_agent(m, p, e) for m, p, e in zip(models, providers, endpoints, strict=True)]


def cmd_run(args) -> None:
    """Handle standard workflow commands."""
    correlation_id = str(uuid.uuid4())[:12]

    try:
        models = args.models if args.models else [args.model]
        providers = args.providers or [args.provider] * len(models)
        endpoints = args.endpoints or [args.url] * len(models)
        agents = build_agents(models, providers, endpoints)

        common = {
            "prompt": args.prompt,
            "image_paths": args.images,
            "detail": args.detail,
            "max_tokens": args.tokens,
            "resize": args.resize,
            "target_size": tuple(args.size),
        }

        if args.workflow == "sequential":
            output = WORKFLOW_REGISTRY["sequential"](agents, **common)

        elif args.workflow == "parallel":
            output = WORKFLOW_REGISTRY["parallel"](agents, **common)

        elif args.workflow == "conditional":
            router_model = args.router_model or agents[0].model
            router_provider = args.router_provider or agents[0].provider
            router_endpoint = args.router_endpoint or agents[0].endpoint
            router = make_agent(router_model, router_provider, router_endpoint)
            if len(args.categories) != len(agents):
                raise ValueError(f"--categories ({len(args.categories)}) must match --models ({len(agents)})")
            output = run_conditional(router, agents, args.categories, **common)

        elif args.workflow == "iterative":
            evaluator = None
            if args.evaluator_model:
                evaluator = make_agent(
                    args.evaluator_model,
                    args.evaluator_provider or agents[0].provider,
                    args.evaluator_endpoint or agents[0].endpoint,
                )
            output = run_iterative(
                agents[0],
                **common,
                evaluator_agent=evaluator,
                max_iterations=args.max_iterations,
            )

        elif args.workflow == "moa":
            agg_model = args.aggregator_model or agents[0].model
            agg_provider = args.aggregator_provider or agents[0].provider
            agg_endpoint = args.aggregator_endpoint or agents[0].endpoint
            aggregator = make_agent(agg_model, agg_provider, agg_endpoint)
            output = run_moa(agents, aggregator, **common)

        elif args.workflow == "react":
            output = run_react(
                agents[0],
                **common,
                enabled_tools=args.tools,
                max_steps=args.max_steps,
            )

        else:
            raise ValueError(f"Unknown workflow: {args.workflow}")

        output["correlation_id"] = correlation_id
        print(json.dumps(output, indent=2))

    except Exception as exc:
        print(json.dumps({"error": str(exc), "correlation_id": correlation_id}, indent=2))
        sys.exit(1)


def cmd_monitor(args) -> None:
    """Handle video monitoring command."""
    correlation_id = str(uuid.uuid4())[:12]

    try:
        # Resolve endpoint
        endpoint = resolve_endpoint(args.provider, args.endpoint)

        # Resolve API key
        env_var = PROVIDER_ENV_MAP.get(args.provider, "OPENAI_API_KEY")
        api_key = os.getenv(env_var, "")
        if not api_key and "localhost" not in endpoint and "127.0.0.1" not in endpoint:
            raise RuntimeError(f"API key not found. Set {env_var} or use --endpoint for a local server.")
        if not api_key:
            api_key = "EMPTY"

        output = run_monitoring(
            video_source=args.video,
            endpoint=endpoint,
            api_key=api_key,
            model=args.model,
            alert_prompt=args.alert_prompt,
            fps=args.fps,
            max_frames=args.max_frames,
            detail=args.detail,
            max_tokens=args.max_tokens,
            continuous=args.continuous,
            interval_seconds=args.interval,
            window_frames=args.window_frames,
            output_jsonl=args.output_jsonl,
        )

        output["correlation_id"] = correlation_id
        print(json.dumps(output, indent=2))

    except Exception as exc:
        print(json.dumps({"error": str(exc), "correlation_id": correlation_id}, indent=2))
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="vlm-agent-gateway",
        description=(
            "VLM Agent Gateway — multi-provider vision inference with "
            "sequential, parallel, conditional, iterative, MoA, ReAct, and monitoring workflows"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================================================
    # Subcommand: run (default workflow execution)
    # ========================================================================
    run_parser = subparsers.add_parser("run", help="Run a VLM workflow")

    # Image / prompt
    run_parser.add_argument("--prompt", "-p", type=str, default="What's in this image?")
    run_parser.add_argument("--images", "-i", type=str, nargs="+", required=True, help="Image paths or URLs")
    run_parser.add_argument("--detail", "-d", type=str, default="low", choices=["auto", "low", "high"])
    run_parser.add_argument("--tokens", "-t", type=int, default=300, help="Max tokens per response")
    run_parser.add_argument("--resize", "-r", action="store_true", help="Resize images with padding")
    run_parser.add_argument("--size", "-s", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))

    # Workflow
    run_parser.add_argument(
        "--workflow",
        "-w",
        type=str,
        default="sequential",
        choices=["sequential", "parallel", "conditional", "iterative", "moa", "react"],
        help="Workflow type (default: sequential)",
    )

    # Agent targets
    run_parser.add_argument("--models", type=str, nargs="+", metavar="MODEL", help="One model per agent")
    run_parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        metavar="PROVIDER",
        help=f"Provider per agent. Known: {list(PROVIDER_ENV_MAP.keys())}",
    )
    run_parser.add_argument("--endpoints", type=str, nargs="+", metavar="URL", help="API endpoint per agent")

    # Single-agent fallback (backward-compatible)
    run_parser.add_argument("--model", "-m", type=str, default="gpt-5.2")
    run_parser.add_argument("--url", "-u", type=str, default=DEFAULT_ENDPOINT)
    run_parser.add_argument("--provider", type=str, default="openai")

    # Special-role agents
    run_parser.add_argument("--aggregator-model", type=str, default=None)
    run_parser.add_argument("--aggregator-provider", type=str, default=None)
    run_parser.add_argument("--aggregator-endpoint", type=str, default=None)
    run_parser.add_argument("--router-model", type=str, default=None)
    run_parser.add_argument("--router-provider", type=str, default=None)
    run_parser.add_argument("--router-endpoint", type=str, default=None)
    run_parser.add_argument("--categories", type=str, nargs="+", default=["general"])
    run_parser.add_argument("--evaluator-model", type=str, default=None)
    run_parser.add_argument("--evaluator-provider", type=str, default=None)
    run_parser.add_argument("--evaluator-endpoint", type=str, default=None)
    run_parser.add_argument("--max-iterations", type=int, default=3)

    # ReAct args
    run_parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=None,
        metavar="TOOL",
        help=f"ReAct tools to enable. Available: {list(BUILTIN_TOOLS.keys())}",
    )
    run_parser.add_argument("--max-steps", type=int, default=5)

    run_parser.set_defaults(func=cmd_run)

    # ========================================================================
    # Subcommand: monitor (video monitoring)
    # ========================================================================
    monitor_parser = subparsers.add_parser("monitor", help="Video monitoring with VLM")

    monitor_parser.add_argument(
        "--video", "-v", required=True, help="Video file path, RTSP URL, or device index (0 for webcam)"
    )
    monitor_parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate (default: 1.0)")
    monitor_parser.add_argument("--max-frames", type=int, default=16, help="Max frames to extract (default: 16)")
    monitor_parser.add_argument("--detail", type=str, default="low", choices=["auto", "low", "high"])
    monitor_parser.add_argument("--max-tokens", type=int, default=1024)

    monitor_parser.add_argument("--provider", "-p", type=str, default="google")
    monitor_parser.add_argument("--model", "-m", type=str, default="gemini-2.5-flash")
    monitor_parser.add_argument("--endpoint", "-e", type=str, default=None)

    monitor_parser.add_argument("--alert-prompt", "-a", required=True, help="Condition to monitor for")
    monitor_parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    monitor_parser.add_argument("--interval", type=float, default=10.0, help="Seconds between cycles (default: 10)")
    monitor_parser.add_argument("--window-frames", type=int, default=8, help="Frames per window (default: 8)")
    monitor_parser.add_argument("--output-jsonl", type=str, default=None, help="Output JSONL file path")

    monitor_parser.set_defaults(func=cmd_monitor)

    # ========================================================================
    # Parse and dispatch
    # ========================================================================
    args = parser.parse_args()

    # Handle legacy mode (no subcommand) for backward compatibility
    if args.command is None:
        # Check if any run-specific args were provided
        if "--images" in sys.argv or "-i" in sys.argv:
            # Legacy mode: parse as run command
            sys.argv.insert(1, "run")
            args = parser.parse_args()
        else:
            parser.print_help()
            sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
