from typing import List, Optional, Callable

import numpy as np
import pandas as pd

#from ..utils import load_external_data
from .dataset_sample import DatasetSample
from .dataset import Dataset
from .molecule_set import MoleculeSet


def _de_duplicate(df: pd.DataFrame, ids: np.ndarray):
    df['id'] = ids
    return df.groupby('id').mean()

def read_protein_peptide_dataset_pandas(protein_df: pd.DataFrame, peptide_df: pd.DataFrame, sample_columns: List[str],
                                        mapping_columns = ['gene'], protein_id_column = 'ProteinID', peptide_id_column = 'Sequence',
                                        protein_mapping_transform: Optional[Callable] = None,
                                        peptide_mapping_transform: Optional[Callable] = None):
    molecules = {}
    values = {}
    for sample_name in sample_columns:
        values[sample_name] = {}
    molecules['protein'] = pd.DataFrame(index=protein_df[protein_id_column].unique())
    protein_values = protein_df.loc[:, [protein_id_column] + sample_columns].rename(columns={protein_id_column:'id'}).set_index('id')
    for sample_name in sample_columns:
        vals = protein_values[[sample_name]].rename(columns={sample_name:'abundance'})
        values[sample_name]['protein'] = vals
    molecules['peptide'] = pd.DataFrame(index=peptide_df[peptide_id_column].unique())
    peptide_values = peptide_df.loc[:, [peptide_id_column] + sample_columns].rename(columns={peptide_id_column:'id'}).set_index('id')
    for sample_name in sample_columns:
            vals = peptide_values[[sample_name]].rename(columns={sample_name:'abundance'})
            values[sample_name]['peptide'] = vals
    mappings = {}
    for mapping_column in mapping_columns:
        mapping = {}
        protein_mapping = pd.DataFrame({'id':protein_df[protein_id_column],
                                         'map_id':protein_df[mapping_column]})
        protein_mapping['map_id'] = protein_mapping['map_id'].str.split(',')
        protein_mapping = protein_mapping.explode('map_id').reset_index(drop=True)
        if protein_mapping_transform is not None:
            mapped = protein_mapping_transform(protein_mapping.map_id)
            protein_mapping = pd.DataFrame(data={'map_id':mapped, 'id':protein_mapping.loc[mapped.index, 'id']})
            protein_mapping = protein_mapping.reset_index(drop=True)
        mapping['protein'] = protein_mapping
        peptide_mapping = pd.DataFrame({'id':peptide_df[peptide_id_column],
                                         'map_id':peptide_df[mapping_column]})
        peptide_mapping['map_id'] = peptide_mapping['map_id'].str.split(',')
        peptide_mapping = peptide_mapping.explode('map_id').reset_index(drop=True)
        if peptide_mapping_transform is not None:
            mapped = peptide_mapping_transform(peptide_mapping.map_id)
            peptide_mapping = pd.DataFrame(data={'map_id':mapped, 'id':peptide_mapping.loc[mapped.index, 'id']})
            peptide_mapping = peptide_mapping.reset_index(drop=True)
        mapping['peptide'] = peptide_mapping
        mappings = {mapping_column: mapping}
    molecule_set = MoleculeSet(molecules=molecules, mappings=mappings)
    dataset = Dataset(molecule_set=molecule_set)
    for sample_name in sample_columns:
        dataset.create_sample(name=sample_name, values=values[sample_name])
    return dataset

def read_dataset_tsv(base_path: str, sample_columns: List[str], 
                     peptide_suffix = '_peptides.tsv', protein_suffix = '_prot.tsv',
                     mapping_column = 'gene', protein_id_column = 'ProteinID', peptide_id_column = 'Sequence'):
    protein_df = pd.read_csv(base_path + protein_suffix, sep='\t')
    peptide_df = pd.read_csv(base_path + peptide_suffix, sep='\t')
    return  read_protein_peptide_dataset_pandas(protein_df=protein_df, peptide_df=peptide_df, sample_columns=sample_columns,
                                 mapping_column=mapping_column, protein_id_column=protein_id_column,
                                 peptide_id_column=peptide_id_column)

# def read_dataset(data_dir_path: str, sample_type: str, load_peptides=True, load_proteins=True, load_mRNA=True, apply_normalization=True,
#                  apply_log: bool=True, write_debug_artifacts:bool = False):
#     abundances = dict()
#     labels = dict()
#     gene_mapping = dict()
#     molecules = dict()

#     if load_proteins:
#         protein_file_name = sample_type + '_prot'
#         loaded = load_external_data.load_protein_data_external(protein_file_name + '.tsv', normalization=apply_normalization, logarithmize=apply_log,
#                                                                file_directory_path=data_dir_path,
#                                                                write_debug_artifacts=write_debug_artifacts)
#         protein_id_list, gene_list_proteins, sample_id_list_proteins, node_abundance_matrix_proteins, node_label_matrix_proteins = loaded
#         abundances['protein'] = _de_duplicate(pd.DataFrame(
#             node_abundance_matrix_proteins, columns=sample_id_list_proteins), protein_id_list)
#         labels['protein'] = _de_duplicate(pd.DataFrame(
#             node_label_matrix_proteins, columns=sample_id_list_proteins), protein_id_list)
#         gene_mapping['protein'] = pd.DataFrame(
#             {'id': protein_id_list, 'map_id': gene_list_proteins})
#         molecules['protein'] = pd.DataFrame(index=abundances['protein'].index)
#     if load_peptides:
#         peptide_file_name = sample_type + '_peptides.tsv'
#         loaded = load_external_data.load_peptide_data_external(peptide_file_name, normalization=apply_normalization, logarithmize=apply_log,
#                                                                file_directory_path=data_dir_path,
#                                                                 write_debug_artifacts=write_debug_artifacts)
#         peptide_id_list, gene_list_peptides, sample_id_list_peptides, node_abundance_matrix_peptides, node_label_matrix_peptides = loaded
#         abundances['peptide'] = _de_duplicate(pd.DataFrame(
#             node_abundance_matrix_peptides, columns=sample_id_list_peptides), peptide_id_list)
#         labels['peptide'] = _de_duplicate(pd.DataFrame(
#             node_label_matrix_peptides, columns=sample_id_list_peptides), peptide_id_list)
#         gene_mapping['peptide'] = pd.DataFrame(
#             {'id': peptide_id_list, 'map_id': gene_list_peptides})
#         molecules['peptide'] = pd.DataFrame(index=abundances['peptide'].index)
#     if load_mRNA:
#         mRNA_file_name = sample_type + '_rna'
#         loaded = load_external_data.load_mRNA_data_external(mRNA_file_name + '.tsv', normalization=apply_normalization, logarithmize=apply_log,
#                                                             file_directory_path=data_dir_path,
#                                                             write_debug_artifacts=write_debug_artifacts)
#         ENTREZID_list, gene_list_mRNA, sample_id_list_mRNA, node_abundance_matrix_mRNA, node_label_matrix_mRNA = loaded
#         abundances['mRNA'] = _de_duplicate(pd.DataFrame(
#             node_abundance_matrix_mRNA, columns=sample_id_list_mRNA), ENTREZID_list)
#         labels['mRNA'] = _de_duplicate(pd.DataFrame(
#             node_label_matrix_mRNA, columns=sample_id_list_mRNA), ENTREZID_list)
#         gene_mapping['mRNA'] = pd.DataFrame(
#             {'id': ENTREZID_list, 'map_id': gene_list_mRNA})
#         molecules['mRNA'] = pd.DataFrame(index=abundances['mRNA'].index)

#     molecule_set = MoleculeSet(molecules=molecules, mappings=gene_mapping)
#     dataset = dict()
#     for key in abundances.keys():
#         for c in abundances[key].columns:
#             if c not in dataset:
#                 dataset[c] = DatasetSample(molecule_set=molecule_set, values=dict(), missing_abundance_value=-100,
#                                        missing_label_value=-100)
#             sample_values = pd.DataFrame(abundances[key][c])
#             sample_values.rename(columns={c: 'abundance'}, inplace=True)
#             sample_values['label'] = labels[key][c]
#             dataset[c].values[key] = sample_values
#     return Dataset(samples=dataset)
