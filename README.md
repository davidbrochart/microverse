# Microverse

In-browser JupyterLab powered by [Jupyverse](https://github.com/jupyter-server/jupyverse).

## Usage

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
- Run `microverse`:
```bash
microverse
```

A server should be running at http://127.0.0.1:8000.
