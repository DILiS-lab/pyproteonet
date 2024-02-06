from typing import Optional

from .missingpy import MissForest
from ..data.dataset import Dataset

def random_forest_impute(
    dataset: Dataset, molecule: str, column: str, molecules_as_variables:bool = False, result_column: Optional[str] = None, **kwargs
) -> Dataset:
    """Impute missing values using a random forest.

    Args:
        dataset (Dataset): Dataset to impute.
        molecule (str): Molecule type to impute.
        column (str): Value column containing the missing values to impute.
        molecules_as_variables (bool, optional): Whether to transpose the input matrix before imputation. Defaults to False.
        result_column (Optional[str], optional): If given, name of the value column to store the imputed values in. Defaults to None.

    Returns:
        Dataset: _description_
    """
    imputer = MissForest(missing_values=dataset.missing_value, **kwargs)
    matrix = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    mask = ~matrix.isna().all(axis=1)

    mat = matrix.loc[mask, :].to_numpy()
    if molecules_as_variables:
        mat = mat.T
    mat = imputer.fit_transform(mat)
    if molecules_as_variables:
        mat = mat.T
    assert mat.shape[0] == mask.sum()
    assert mat.shape[1] == matrix.shape[1]

    matrix.loc[mask, :] = mat
    col_means = matrix.mean()
    for c in matrix:
        col = matrix[c]
        col[col.isna()] = col_means[c]

    if result_column is not None:
        dataset.set_wf(molecule=molecule, column=result_column, matrix=matrix)
    
    vals = matrix.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals