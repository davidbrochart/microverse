# Microverse

In-browser JupyterLab powered by [Jupyverse](https://github.com/jupyter-server/jupyverse).

## Local deployment

- Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
- Create an environment and activate it:
```bash
micromamba create -n microverse
micromamba activate microverse
```
- Install `pip` and `empack`:
```bash
micromamba install pip empack
```
- Install `microverse`:
```bash
pip install -e .
```
- Run `microverse`:
```bash
microverse --serve
```

A server should be running at http://127.0.0.1:8000.

## GitHub pages deployment

The `main` branch is deployed on GitHub pages at [https://davidbrochart.github.io/microverse/](https://davidbrochart.github.io/microverse/).
