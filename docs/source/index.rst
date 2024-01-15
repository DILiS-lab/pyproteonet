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
   :caption: Python API

   api/data
   api/simulation
   api/aggregation
   api/imputation
   api/normalization


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
