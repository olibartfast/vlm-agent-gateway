# VLM Agent Gateway

Experimenting with multi-provider Vision Language Model inference using sequential, parallel, conditional, iterative, Mixture-of-Agents (MoA), and ReAct (Reasoning + Acting) agent workflow patterns. Dispatch to one or more VLM providers as chained, parallel, routed, looping, or tool-using agent pipelines from a single CLI call.

## Main Project

### VLM Agent Gateway (`vlm-agent-gateway/`)
A multimodal agent client implementing the full set of agent workflow patterns across multiple providers.

**[📖 Source](vlm-agent-gateway/main.py)**

#### Key Features
- 🤖 **6 workflow modes** — `sequential`, `parallel`, `conditional`, `iterative`, `moa`, `react` via `--workflow`
- 🔗 **Sequential chaining** — each agent receives prior agents' outputs as accumulated context
- ⚡ **Parallel fan-out** — same input dispatched concurrently, best answer selected
- 🔀 **Conditional routing** — router agent classifies input, specialist agent handles request
- 🔁 **Iterative refinement** — output fed back as context each iteration until convergence
- 🧩 **Mixture-of-Agents** — parallel proposers → aggregator synthesizes final answer
- 🛠️ **ReAct** — agent reasons, selects a tool, observes result, loops until Final Answer
- 🔌 **Multi-provider** — OpenAI, Anthropic, Google, Together, Groq, Mistral, Azure per-agent
- 🔍 **Observability** — per-agent latency, `correlation_id`, `stop_reason`, raw responses

#### Workflow Quick Start

**Sequential** — each stage builds on the previous (default):
```bash
python vlm-agent-gateway/main.py \
    --workflow sequential \
    --prompt "Describe this image" \
    --images image.jpg \
    --models gpt-4o-mini gpt-4o \
    --providers openai openai
```

**Parallel** — same input to multiple agents, best answer selected:
```bash
python vlm-agent-gateway/main.py \
    --workflow parallel \
    --prompt "What objects are in this image?" \
    --images image.jpg \
    --models gpt-4o-mini claude-3-haiku \
    --providers openai anthropic \
    --endpoints https://api.openai.com/v1/chat/completions https://api.anthropic.com/v1/messages
```

**Conditional** — router classifies input, specialist handles it:
```bash
python vlm-agent-gateway/main.py \
    --workflow conditional \
    --prompt "Analyze this image" \
    --images image.jpg \
    --models gpt-4o-mini gpt-4o-mini \
    --providers openai openai \
    --categories document scene \
    --router-model gpt-4o-mini \
    --router-provider openai
```

**Iterative** — refines answer across iterations with context accumulation:
```bash
python vlm-agent-gateway/main.py \
    --workflow iterative \
    --prompt "Describe this image in detail" \
    --images image.jpg \
    --model gpt-4o-mini \
    --max-iterations 3 \
    --evaluator-model gpt-4o \
    --evaluator-provider openai
```

**Mixture-of-Agents** — parallel proposers → aggregator synthesizer:
```bash
python vlm-agent-gateway/main.py \
    --workflow moa \
    --prompt "Describe this image in detail" \
    --images image.jpg \
    --models gpt-4o-mini claude-3-haiku \
    --providers openai anthropic \
    --endpoints https://api.openai.com/v1/chat/completions https://api.anthropic.com/v1/messages \
    --aggregator-model gpt-4o \
    --aggregator-provider openai
```

**ReAct** — agent reasons, picks a tool, observes output, loops to Final Answer:
```bash
python vlm-agent-gateway/main.py \
    --workflow react \
    --prompt "How many people are in this image and what are they doing?" \
    --images image.jpg \
    --model gpt-4o \
    --tools describe detect_objects count_objects \
    --max-steps 5
```

#### Workflow Modes

| Mode | Agents | Data flow |
|------|--------|-----------|
| `sequential` | ≥ 1 | Agent-1 → output-1 → Agent-2 (with context) → … → final |
| `parallel` | ≥ 2 | All agents receive same input concurrently → best answer selected |
| `conditional` | ≥ 2 | Router classifies input → matching specialist handles request |
| `iterative` | 1 + optional evaluator | Agent loops, feeding output back as context until convergence |
| `moa` | ≥ 2 + aggregator | Parallel proposers → aggregator synthesizes all candidates |
| `react` | 1 | Thought → Action (tool) → Observation loop until Final Answer |

#### Built-in ReAct Tools

| Tool | Description |
|------|-------------|
| `describe` | Detailed image description with optional focus prompt |
| `detect_objects` | List all visible objects as a JSON array |
| `read_text` | Extract visible text (OCR) |
| `analyze_region` | Focus on a described region and answer a question about it |
| `count_objects` | Count occurrences of a specific named object |

Omit `--tools` to enable all tools. Pass a subset to restrict the agent's action space.

#### Supported Providers

| Provider | `--providers` value | API key env var |
|----------|--------------------|-----------------| 
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Google | `google` | `GOOGLE_API_KEY` |
| Together AI | `together` | `TOGETHER_API_KEY` |
| Azure OpenAI | `azure` | `AZURE_OPENAI_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| Mistral | `mistral` | `MISTRAL_API_KEY` |

---

### C++ Client
A C++ CLI for single-branch VLM inference using OpenAI-compatible APIs.

**[📖 Full Documentation](vlm-inference-client/cpp/Readme.md)**

#### Key Features
- 🔌 Multiple API provider support (OpenAI, Together, vLLM, and more)
- 🖼️ Multimodal capabilities (text + multiple images)
- 🔄 Automatic image preprocessing and resizing
- ⚙️ Configurable detail levels and token limits
- 🌐 Support for local files and image URLs

#### Quick Start
```bash
./vlm-inference-client \
    --prompt "Describe this image" \
    --images image.jpg \
    --model gpt-4o \
    --api_endpoint https://api.openai.com/v1/chat/completions \
    --api_key_env OPENAI_API_KEY
```

---

## Additional Resources


## Other Documentation/Resources

- **[Benchmarks](docs/benchmarks.md)** - VLM evaluation benchmarks and leaderboards
- **[Courses & Tutorials](docs/courses.md)** - Online courses and learning resources
- **[API Services](docs/api-services.md)** - Vision multimodal API providers
- **[Finetuning](docs/finetuning.md)** - Resources for finetuning VLMs
- **[RAG](docs/rag.md)** - Multimodal RAG resources
- **[Inference](docs/inference.md)** - Inference frameworks and tools
- **[Cloud GPU](docs/cloud-gpu.md)** - GPU rental services
- **[Google AI](docs/google.md)** - Google-specific resources (Gemini, Vertex AI)
- **[Llama](docs/llama.md)** - Llama-specific resources
- **[Nvidia Dynamo](https://developer.nvidia.com/dynamo)** - Nvidia framework that serves VLM/LLM models by wrapping tensort-llm/vllm/sglang backends
