# maxplotlib

This is a wrapper for matplotlib so I can produce figures with consistent formatting. It also has some pretty nice additions such as using layers and exporting to tikz.

## Install

Create and activate python environment

```
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the code and requirements with pip

```
pip install -e .
```

Additional dependencies for developers can be installed with

```
pip install -e ".[dev]"
```

Some examples can be found in `tutorials/`