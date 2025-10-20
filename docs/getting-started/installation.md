# Installation

## Requirements

- Python 3.12 or higher
- pip or uv package manager

## Install from PyPI

```bash
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ causalpype==0.0.1
```

## Install from Source

```bash
git clone https://github.com/palimisis/causalpype.git
cd causalpype
pip install -e .
```

## Verify Installation

```python
import causalpype
print(causalpype.__version__)
```

