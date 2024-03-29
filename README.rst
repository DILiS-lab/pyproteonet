PyProteoNet
===========

**PyProteoNet** is a Python package for imputation, peptide-to-protein aggregation 
as well as in silico data creation of proteomics data with 
a specific focus on the development and evaluation of imputation methods for proteomics.
Mass spectrometry experiments are represented by a set of molecules.
Generally, those molecules are proteins and peptides. 
However, the underlying data structures allow for the extension to arbitrary molecule types
like mRNA.
Given a dataset, PyProteoNet provides functions for peptide-to-protein aggregation as well as a range of common imputation methods.
Results from multiple aggregation and imputation methods can be stored for the same dataset facilitating evaluation and comparison of different methods.

Next to established imputation methods PyProteoNet also implements our newly proposed Graph Neural Network imputation
which operates on the protein-peptide graph and, therefore, jointly takes proteins and peptides into account.

Read the documentation for more details: `https://pyproteonet.readthedocs.io <https://pyproteonet.readthedocs.io>`_.

Installation
============
First, clone the PyProteoNet repository to a local directory.
Installation is best done within a Conda environment (you might also consider Mamba as a drop-in replacement for Conda with higher performance).
You can install an already configured environment using the environment.yml included in the repository. To do so, just run the following command inside your cloned PyProteoNet directory.
This creates a mamba environment called pyproteonet:

``mamba env create --name pyproteonet --file environment.ym``

Alternatively, you can create your own custom environment or use an existing one. In this case, it is advised to install the following requirements via Conda/Mamba because they are either not available via pip or using the pip version might lead to problems:

* r-base, ``conda install -c conda-forge r-base`` (you need an R installation because several provided imputation methods are wrappers around R packages)
* Pytorch, ``conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`` (see the `PyTorch website <https://pytorch.org/get-started/locally/>`_) for instructions for your system and Cuda version) 
* DGL, ``conda install -c dglteam dgl`` (see the `DGL website <https://www.dgl.ai/pages/start.html>`_) for instructions for your system and Cuda version)

Afterward, you can install PyProteoNet in the created Conda/Mamba environment via pip by running the following command inside the root of your PyProtoNet directory:

``pip install ./``

Getting Started and Overview
============================
There is a `Getting Started Notebook <https://github.com/Tobias314/pyproteonet/blob/main/docs/source/notebooks/getting_started.ipynb>`_ which gives a first overview of how PyProteoNet can be used

Documentation
============================
There is a `Read the Docs <https://pyproteonet.readthedocs.io/en/latest>`_ page containing more examples and documentation.


References
==========
The implementation of autoencoder-based as well as collaborative-filtering-based imputation methods is taken from the `PIMMS project <https://github.com/RasmussenLab/pimms>`_.
The implementation of MaxLFQ is a slightly modified version of the implementation of `DPKS <https://github.com/InfectionMedicineProteomics/DPKS/>`_.
