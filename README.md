# Introduction

This repository contains all the materials used in the experiments described in the paper "[Explainable Fault Classification and Severity Diagnosis in Rotating Machinery Using Kolmogorov–Arnold Networks](https://www.mdpi.com/1099-4300/27/4/403)".

In the study, we augmented the [CWRU](https://engineering.case.edu/bearingdatacenter) and [MaFaulDa](https://www02.smt.ufrj.br/~offshore/mfs) datasets and subsequently extracted feature libraries from them, in order to apply our Kolmogorov-Arnold network (KAN)-based framework for automatic feature and model selection. The experiments focused on three key tasks: fault detection, fault classification, and severity classification. While centered on bearing faults, our approach also extended to other types of machinery failures via the MaFaulDa dataset.


# Files

The extracted feature libraries for each dataset can be found in the [data](/data/) folder. The [pykan](/pykan/) folder contains some modified files from the [pykan library](https://github.com/KindXiaoming/pykan) (see below for details). The `utils.py` file contains some basic utilities used throughout all experiments. The main files are `CWRU.ipynb` and `MaFaulDa.ipynb`, containing the structure to carry out the experiments presented in the paper.

# Installation

We advise creating a virtual environment, for instance via `venv` as

```
python3 -m venv env
```

Then, activate the environment via `env\Scripts\activate` for Windows or `source env/bin/activate` for Linux and install dependencies via

```
pip3 install -r requirements.txt
```

Note that packages like Jupyter Lab are not included in the requirements, although the experiments are in `.ipynb` notebooks, so you will have to install one to run them, e.g. via `pip3 install jupyterlab`. Additionally, `torch` is not included in the dependencies because we used the GPU version for acceleration. Depending on your CUDA setup, you are advised to install the corresponding version via [their website](https://pytorch.org/).

As mentioned above, our code runs on a modified version of the pykan library. Providing an extended changelog would be less practical, so instead we include our versions of the modified files in the pykan folder. You may either install pykan via pip (which is also not included in the dependencies) and then replace the corresponding files, or directly download their codebase locally and replace the corresponding files.

During development, our version of torch was `2.5.0+cu118` and our version of pykan was `v0.2.7`.


# Attribution

If the code and/or extracted feature libraries presented here helped you for your own work, feel free to cite our GitHub repo and/or paper as:

```
@article{explainfault,
      author = {Spyros Rigas and Michalis Papachristou and Ioannis Sotiropoulos and Georgios Alexandridis},
      title = {Explainable Fault Classification and Severity Diagnosis in Rotating Machinery Using Kolmogorov–Arnold Networks},
      journal = {Entropy},
      volume = {27},
      year = {2025},
      number = {4},
      doi = {10.3390/e27040403}
}
```
