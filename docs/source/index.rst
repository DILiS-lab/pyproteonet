.. PyProteoNet documentation master file

Welcome to PyProteoNet's documentation!
=======================================

**PyProteoNet** is a Python package for imputation, peptide-to-protein aggregation 
as well as in silico data creation of proteomics data with 
a specific focus on the development and evaluation of imputation methods for proteomics.
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

Features
========
Data
----
Datastructures for proteomics data consisting of per-molecule values (e.g. abundance measurements) for several samples and different interrelated molecule types (e.g. proteins and peptides).
Pandas interface for easy data handling and manipulation.

Simulation
----------
Several functions for the creation of in-silico proteomics data. This includes the simulation of protein and peptide abundances, the simulation of fold changes and the simulation of missing values.

Aggregation
-----------
Native implementations of MaxLFQ, IBAQ, Top3
as well as simple peptide-to-protein aggregation methods like sum, mean, median.

Imputation
----------
Several reference imputation methods are provided either implemented directly in Python or via a wrappers around common R packages.

+---------------------------------+------------------------------------------------+-------------------------------------+
| Single Value Imputation Methods | Global Structure Imputation Methods            | Local Similarity Imputation Methods |
+=================================+================================================+=====================================+
| MinDet                          | PCA, PPCA, BPCA                                | MissForest                          |
+---------------------------------+------------------------------------------------+-------------------------------------+
| MInProb                         | ISVD                                           | KNN                                 |
+---------------------------------+------------------------------------------------+-------------------------------------+
| Mean, Median                    | Denoising Autoencoder, Variational Autoencoder | Collaborative Filtering             |
+---------------------------------+------------------------------------------------+-------------------------------------+
|                                 | MLE                                            | Local Least Squares                 |
+---------------------------------+------------------------------------------------+-------------------------------------+
|                                 |                                                | Iterative Imputation                |
+---------------------------------+------------------------------------------------+-------------------------------------+
Next to those established methods, PyProteoNet provides imputation methods based on homogeneous as well as heterogeneous graph neural networks (GNNs) specifically for the imputation of missing values in proteomics data.
New DNN- and GNN-based methods can be easily implemented and integrated into the existing framework.

Evaluation
----------
Several metrics for the evaluation of aggregation and imputation methods.
Common evaluation metrics like abosulte error (AE) mean absolute error (MAE), mean squared error (MSE), Pearson correlation coefficient (PearsonR)
can be computed for comparing aggregated/imputed against ground truth abundance values.
In addition, catering for the specific needs of proteomics data fold-change-based evaluation methods are provided to evaluate based on molecule ratios between samples as well as discovered differential expressions. 

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/getting_started
   notebooks/simulation
   notebooks/evaluate_imputation_abundance
   notebooks/evaluate_imputation_fold_change
   notebooks/imputation_method_development

.. toctree::
   :maxdepth: 1
   :caption: Python API

   api/data
   api/aggregation
   api/simulation
   api/imputation
   api/metrics
   api/io
   api/masking
   api/dgl


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
