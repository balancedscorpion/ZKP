# Zero-Knowledge Proof Demo

This repository contains a Jupyter notebook tutorial demonstrating various concepts of zero-knowledge proofs through interactive examples.

## Overview

Zero-knowledge proofs are cryptographic methods that allow one party to prove to another that they know a value without revealing the actual value. This tutorial explores:

- Basic principles of zero-knowledge proofs
- Interactive verification protocols
- Graph coloring as a zero-knowledge proof example
- Practical implementation details with visualizations

## Requirements

- Python 3.10 or higher
- `uv` (Python package installer and environment manager)
- Git

## Installation

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/balancedscorpion/zero-knowledge-proof.git
cd zero-knowledge-proof
```

## Getting Started with `uv`

[`uv`](https://github.com/astral-sh/uv) is a fast Python package installer and environment manager. Here's how to set up and run the tutorial:

### 1. Install `uv` (if you haven't already)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create a virtual environment

```bash
uv venv
```

This will create a virtual environment in a `.venv` directory in the current folder.

### 3. Activate the virtual environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Install the dependencies

```bash
uv pip install .
```

This will install all dependencies specified in the `pyproject.toml` file.

### 5. Start the Jupyter notebook

```bash
jupyter notebook
```

This will open a browser window with the Jupyter notebook interface. Navigate to the tutorial notebook and open it to begin.

## Tutorial Contents

The tutorial is organized into the following sections:

1. Introduction to Zero-Knowledge Proofs
2. Basic Interactive Proofs
3. Graph Coloring ZKP Example
4. Practical Applications
5. Hands-on Exercises

## Contributing

Contributions to improve the tutorial or add new examples are welcome! Please feel free to submit a pull request.

## Attribution

Created by [balancedscorpion](https://github.com/balancedscorpion)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
