.. PyProteoNet documentation master file

Welcome to PyProteoNet's documentation!
=======================================

**PyProteoNet** is a Python package for protein quantification 
and imputation as well as in silico data generation.
Mass spectrometry experiments are represented by a set of molecules.
Generally those molecules are proteins and peptides. 
However, the underlying data structures allow for arbitrary molecule types
such that additional measurements like mRNA can also be supported.
Molecules and their relations are represented by a graph structure.
For example for a regular MS experiment measuring peptide abundances
which are then aggregated into protein abundances this results in a graph with proteins
and peptides as nodes where every peptide is connected via an edge to all proteins it can
be found in.
One big advantage of this graph structure is the quantifiction and imputation of proteins
by taking peptide-to-protein relations into account. From a technical point of view this
is achived by training a graph neural network (GNN)
for protein quantification and imputation. 


Installation
============

Installation is best done within a Conda environment (you might also consider Mamba as a drop-in replacement for Conda with higher performance). 
It is advised to install the following requirements via Conda/Mamba because they are either not available via pip or using the pip version might lead to problems:

* r-base, ``conda install -c conda-forge r-base`` (you need an R installation because several provided imputation methods are wrappers around R packages)
* Pytorch, ``conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`` (see the `PyTorch website <https://pytorch.org/get-started/locally/>`_ for instructions for your system and Cuda version) 
* DGL, ``conda install -c dglteam dgl`` (see the `DGL website <https://www.dgl.ai/pages/start.html>`_ for instructions for your system and Cuda version)

Afterward you can clone PyProteoNet from github (`https://github.com/Tobias314/pyproteonet <https://github.com/Tobias314/pyproteonet>`_) and install it in the created Conda/Mamba environment via pip by running the following command inside the root of the pyprotonet directory:
``pip install ./``


.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/getting_started
   notebooks/simulation
   notebooks/evaluate_imputation_abundance
   notebooks/evaluate_imputation_fold_change

.. toctree::
   :maxdepth: 10
   :caption: Python API

   api/data
   api/aggregation
   api/simulation
   api/imputation
   api/metrics


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
