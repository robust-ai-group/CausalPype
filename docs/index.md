# CausalPype

A Python library for building modular causal inference pipelines.

## Overview

CausalPype provides a simple, step-based framework for causal analysis workflows. Instead of writing scattered scripts, you define reusable pipeline steps that can be chained together.

## Key Features

- **Modular Architecture** - Build pipelines from composable steps
- **Flexible Configuration** - Define pipelines in Python or YAML
- **Extensible** - Easy to create custom steps for your needs
- **Causal Models** - Built on DoWhy for structural causal modeling

## Quick Example

```python
from causalpype import Pipeline, PipelineConfig

config = PipelineConfig(outcome='y', persist=True)
pipeline = Pipeline(config)

results = (pipeline
    .load_data('data.csv')
    .train_model(graph=graph)
    .evaluate()
    .run())
```

## Installation

```bash
pip install causalpype
```

## Current Status

!!! warning
    **Early Development** - API may change. Not recommended for production use yet.

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Core Concepts](guide/concepts.md)
