# Project AWARENESS

A fully autonomous, on-device Personal AI Agent for Termux/Android in Python.

## Overview

Project AWARENESS is a privacy-first, user-sovereign autonomous AI agent system designed for mobile deployment. It features a hybrid multi-agent architecture with intelligent resource management, security sandboxing, and adaptive learning capabilities.

## Architecture

- **Kernel**: Stateless asyncio entry point with multiprocessing orchestration
- **RouterAgent**: Central nervous system managing task state machines
- **MemoryControllerAgent**: Three-tier memory hierarchy (Hot/Warm/Cold)
- **ContextAgent**: Serialized context management with compression
- **SecurityAgent**: Trust-based permission enforcement
- **LearningAgent**: Metacognitive background learning loop

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Performance Targets

- Idle state: <1% CPU, <20MB RAM
- Zero security breaches
- Rising Trust Score through collaborative evolution