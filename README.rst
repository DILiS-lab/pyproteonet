  ⚠️ **Warning**: This package is still under heavy development!


PyProteoNet
===========

**PyProteoNet** is a Python package for protein aggregation 
and imputation, as well as in silico data generation.
Mass spectrometry experiments are represented by a set of molecules.
Generally, those molecules are proteins and peptides. 
However, the underlying data structures allow for the extension to arbitrary molecule types
like mRNA.
Given a dataset PyProteoNet provides functions for peptide to protein aggregation as well as a range of common imputation methods.
Results from multiple aggregation and imputation methods can be stored for the same dataset facilitating evaluation and comparison of different methods.

Next to established imputation methods PyProteoNet also implements our newly proposed Graph Neural Network imputation
which operates on the protein-peptide graph and, therefore, jointly takes proteins and peptides into account.

Installation
============

Installation is best done within a Conda environment (you might also consider Mamba as a drop-in replacement for Conda with higher performance). 
It is advised to install the following requirements via Conda/Mamba because they are either not available via pip or using the pip version might lead to problems:

* r-base, ``conda install -c conda-forge r-base`` (you need an R installation because several provided imputation methods are wrappers around R packages)
* Pytorch, ``conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`` (see the `PyTorch website <https://pytorch.org/get-started/locally/>`_) for instructions for your system and Cuda version) 
* DGL, ``conda install -c dglteam dgl`` (see the `DGL website <https://www.dgl.ai/pages/start.html>`_) for instructions for your system and Cuda version)

Afterward you can clone and install pyproteonet in the created Conda/Mamba environment via pip by running the following command inside root the pyprotonet directory:
``pip install ./``

Getting Started and Overview
============================
There is a `Getting Started Notebook <https://github.com/Tobias314/pyproteonet/blob/main/docs/source/notebooks/getting_started.ipynb>`_ which gives a first overview of how PyProteoNet can be used
