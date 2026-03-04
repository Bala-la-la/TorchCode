# 🔥 TorchCode — PyTorch 手写算子训练场

A self-hosted Jupyter-based environment for practicing PyTorch operator implementations. Like LeetCode, but in a real notebook.

## Quick Start

```bash
make run
```

Open **http://localhost:8888** — that's it.

## How It Works

Each problem has **two** notebooks:

| File | Purpose |
|------|---------|
| `01_relu.ipynb` | Blank template — write your code here |
| `01_relu_solution.ipynb` | Reference solution — check when stuck |

**Blank templates reset on every `make run`** so you always start fresh.
If you want to keep your code, save it under a different filename.

### Workflow

1. Open a blank notebook (e.g. `01_relu.ipynb`)
2. Read the problem description
3. Implement your solution in the marked cell
4. Debug freely — `print(x.shape)`, check gradients, whatever you need
5. Run the judge cell: `check("relu")`
6. See instant colored feedback with per-test results
7. Stuck? Open `01_relu_solution.ipynb` or call `hint("relu")`

## Problems

### Basics
| # | Problem | Difficulty |
|---|---------|-----------|
| 1 | Implement ReLU | 🟢 Easy |
| 2 | Implement Softmax | 🟢 Easy |
| 3 | Simple Linear Layer | 🟡 Medium |
| 4 | Implement LayerNorm | 🟡 Medium |
| 7 | Implement BatchNorm | 🟡 Medium |
| 8 | Implement RMSNorm | 🟡 Medium |

### Attention Mechanisms
| # | Problem | Difficulty |
|---|---------|-----------|
| 5 | Softmax Attention | 🔴 Hard |
| 9 | Causal Self-Attention | 🔴 Hard |
| 6 | Multi-Head Attention | 🔴 Hard |
| 10 | Grouped Query Attention | 🔴 Hard |
| 11 | Sliding Window Attention | 🔴 Hard |
| 12 | Linear Self-Attention | 🔴 Hard |

### Full Architecture
| # | Problem | Difficulty |
|---|---------|-----------|
| 13 | GPT-2 Transformer Block | 🔴 Hard |

## Commands

```bash
make run    # Build & start (http://localhost:8888)
make stop   # Stop the container
make clean  # Stop + clear progress
```

## In-Notebook Commands

```python
from torch_judge import check, hint, status

status()                    # Progress dashboard
check("relu")               # Judge your implementation
hint("causal_attention")    # Get a hint
```

## Architecture

```
┌──────────────────────────────────────────┐
│           Docker Container               │
│                                          │
│  JupyterLab (:8888)                     │
│    ├── templates/  (baked in image)      │
│    ├── solutions/  (baked in image)      │
│    ├── torch_judge/ (judge engine)       │
│    └── PyTorch (CPU)                     │
│                                          │
│  entrypoint.sh:                          │
│    1. Copy templates → notebooks/ (reset)│
│    2. Copy solutions → notebooks/        │
│    3. Start JupyterLab                   │
│                                          │
│  Volumes:                                │
│    ./notebooks  (working dir, persists)  │
│    ./data       (progress.json)          │
└──────────────────────────────────────────┘
```

Single container. Single port. No database. No frontend framework.
