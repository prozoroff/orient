# Orienteering Route Planning Pipeline

Python pipeline for processing orienteering maps: semantic segmentation,
control-point detection, terrain-aware pathfinding, and score-orienteering route
planning.

## Repository contents

- `src/` — segmentation, control detection, routing, and route optimization;
- `scripts/` — command-line utilities for inference and diagnostics;
- `notebooks/` — cleaned examples without saved cell outputs or credentials;
- `configs/` — model and inference configuration files.

Datasets, model checkpoints, generated outputs, caches, logs, images, videos,
and deployment code are intentionally not included.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install a compatible PyTorch build first using the instructions from
the official PyTorch website. For notebook environments with PyTorch already
installed, use `requirements-cloud.txt`.

## Configuration

Copy the environment template and provide credentials only on your machine:

```bash
cp .env.example .env
```

The VLM control detector supports Yandex AI Studio and OpenAI. Never commit the
resulting `.env` file. Model checkpoints must be supplied separately and their
paths passed to the relevant notebook or command-line utility.

## Typical pipeline

The high-level entry point is `src.orienteering.plan_course`. It combines:

1. control and start/finish detection;
2. semantic segmentation of the map;
3. conversion of terrain classes into a traversal-cost grid;
4. route-time matrix construction;
5. score-orienteering optimization;
6. reconstruction and visualization of route legs.

See `notebooks/course_plan_demo.ipynb` for an end-to-end example and the other
notebooks for individual pipeline stages.

## Data and weights

This repository contains code only. You need to provide your own maps, labels,
and trained model checkpoint. The default training and inference settings are in
`configs/`.
