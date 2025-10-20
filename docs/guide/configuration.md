# Pipeline Configuration

Configuration controls pipeline behavior and step parameters.

## PipelineConfig

```python
from causalpype import PipelineConfig

config = PipelineConfig(
    run_id='my_analysis',     # Unique identifier
    outcome='y',              # Target variable
    persist=True,             # Save outputs to disk
    output_dir='results',     # Where to save
    random_seed=42            # For reproducibility
)
```

## Reporting Configuration

Control what gets saved in outputs:

```python
pipeline.disable_reporting()  # No outputs
pipeline.enable_reporting()   # All outputs (default)
pipeline.disable_figures()    # Skip plots
```

More details coming soon.
