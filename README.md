# VLM Agent Gateway

Multi-provider Vision Language Model inference framework with 7 workflow patterns: sequential, parallel, conditional, iterative, Mixture-of-Agents (MoA), ReAct (Reasoning + Acting), and video monitoring.

## Installation

```bash
# Install from source
pip install -e .

# With video monitoring support (requires OpenCV)
pip install -e ".[video]"

# With development tools
pip install -e ".[dev,video]"
```

## Quick Start

```bash
# Sequential workflow (default) - each stage builds on previous
vlm-agent-gateway run --workflow sequential \
    --prompt "Describe this image" \
    --images image.jpg \
    --models gpt-5-mini gpt-5 \
    --providers openai openai

# Parallel workflow - same input to multiple agents
vlm-agent-gateway run --workflow parallel \
    --prompt "What objects are in this image?" \
    --images image.jpg \
    --models gpt-5-mini Qwen/Qwen3.5-VL-72B-Instruct \
    --providers openai together

# ReAct workflow - agent reasons and uses tools
vlm-agent-gateway run --workflow react \
    --prompt "Count the people and describe what they're doing" \
    --images image.jpg \
    --model gpt-5 \
    --tools describe detect_objects count_objects

# Video monitoring - continuous or single-shot
vlm-agent-gateway monitor \
    --video ./sample.mp4 \
    --alert-prompt "Is anyone falling or in distress?" \
    --provider google \
    --model gemini-3-flash

# Continuous webcam monitoring
vlm-agent-gateway monitor \
    --video 0 \
    --alert-prompt "Has anyone entered the restricted area?" \
    --continuous --interval 10
```

## Features

- **7 workflow patterns** — `sequential`, `parallel`, `conditional`, `iterative`, `moa`, `react`, `monitor`
- **Multi-provider support** — OpenAI, Anthropic, Google, Together, Azure, Groq, Mistral
- **Video monitoring** — Fall detection, security monitoring, safety compliance
- **ReAct tools** — describe, detect_objects, read_text, analyze_region, count_objects
- **Observability** — per-agent latency, correlation IDs, structured JSON output

## Workflow Modes

| Mode | Agents | Data Flow |
|------|--------|-----------|
| `sequential` | ≥ 1 | Agent-1 → output-1 → Agent-2 (with context) → … → final |
| `parallel` | ≥ 2 | All agents receive same input concurrently → best answer |
| `conditional` | ≥ 2 | Router classifies input → matching specialist handles |
| `iterative` | 1 + evaluator | Agent loops, feeding output back until convergence |
| `moa` | ≥ 2 + aggregator | Parallel proposers → aggregator synthesizes |
| `react` | 1 | Thought → Action (tool) → Observation loop |
| `monitor` | 1 | Video frames → VLM analysis → structured alerts |

## Video Monitoring

The `monitor` command supports video-capable VLMs for real-time monitoring:

```bash
# Fall detection
vlm-agent-gateway monitor \
    --video ./elderly_room.mp4 \
    --alert-prompt "Is anyone falling, lying on the floor, or in distress?" \
    --fps 1 --max-frames 30

# Continuous security monitoring
vlm-agent-gateway monitor \
    --video rtsp://camera.local:554/stream \
    --alert-prompt "Has anyone entered the restricted zone?" \
    --continuous --interval 10 --window-frames 8

# Self-hosted with vLLM
vlm-agent-gateway monitor \
    --video 0 \
    --endpoint http://localhost:8000/v1/chat/completions \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --alert-prompt "Detect any hazard" \
    --continuous
```

See [docs/video-vlm-agents.md](docs/video-vlm-agents.md) for model recommendations and deployment guides.

## Supported Providers

| Provider | `--provider` | API Key Env Var |
|----------|--------------|-----------------|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Google | `google` | `GOOGLE_API_KEY` |
| Together AI | `together` | `TOGETHER_API_KEY` |
| Azure OpenAI | `azure` | `AZURE_OPENAI_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| Mistral | `mistral` | `MISTRAL_API_KEY` |

## Python API

```python
from vlm_agent_gateway import run_sequential, run_react, run_monitoring
from vlm_agent_gateway.cli import make_agent

# Create agents
agent = make_agent("gpt-5", "openai", "https://api.openai.com/v1/chat/completions")

# Run workflow
result = run_react(
    agent=agent,
    prompt="Describe this image and count the people",
    image_paths=["image.jpg"],
    detail="low",
    max_tokens=500,
    resize=False,
    target_size=(512, 512),
    enabled_tools=["describe", "count_objects"],
    max_steps=5,
)
print(result["content"])
```

## C++ Client

A lightweight C++ client for single-shot OpenAI-compatible inference.

```bash
cd vlm-inference-client/cpp
mkdir build && cd build
cmake .. && make

./vlm-inference-client \
    --prompt "Describe this image" \
    --images image.jpg \
    --model gpt-4o \
    --api_endpoint https://api.openai.com/v1/chat/completions \
    --api_key_env OPENAI_API_KEY
```

See [vlm-inference-client/cpp/Readme.md](vlm-inference-client/cpp/Readme.md) for details.

## Documentation

- [Video VLM Agents Guide](docs/video-vlm-agents.md) - Video-capable VLMs, vLLM deployment, hardware sizing
- [API Services](docs/api-services.md) - Vision multimodal API providers
- [Benchmarks](docs/benchmarks.md) - VLM evaluation benchmarks
- [Inference](docs/inference.md) - Inference frameworks and tools

## License

MIT
