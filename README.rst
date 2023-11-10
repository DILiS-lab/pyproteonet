PyProteoNet
===========

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
It is advised to install the following requirements via Conda/Mamba because installing them via pip might lead to problems or because they are not available via pip:

* r-base, ``conda install -c conda-forge r-base`` (you need an R installation because several provided imputation methods are wrappers around R packages)
* Pytorch, ``conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`` (see [the PyTorch website](https://pytorch.org/get-started/locally/) for instructions for your system and Cuda version) 
* DGL, ``conda install -c dglteam dgl`` (see the [DGL website](https://www.dgl.ai/pages/start.html) for instructions for your system and Cuda version)

Afterward you can clone and install pyproteonet in the created Conda/Mamba environment via pip by running the following comman inside root the pyprotonet directory:
``pip install ./``
