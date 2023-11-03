from typing import Optional, Union

import numpy as np
import scipy

from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset
from ..quantification.flyability import estimate_flyability_upper_bound
from .protein_peptide import simulate_protein_peptide_dataset
from .missing_values import simulate_mcars, simulate_mnars_thresholding
from .utils import get_numpy_random_generator


class ProteinPeptideDatasetMocker:
    def __init__(
        self,
        dataset: Dataset,
        mapping: str,
        column: str,
        protein_molecule: str = "protein",
        peptide_molecule: str = "peptide",
        noise_quantile: float = 0.01,
    ):
        self.dataset = dataset
        self.protein_molecule = protein_molecule
        self.peptide_molecule = peptide_molecule
        self.column = column
        self.mapping = mapping
        self.log_abundance_mean = (
            np.log(self.dataset.values[self.peptide_molecule][self.column]).groupby("id").mean().mean()
        )
        self.log_abundance_std = (
            np.log(self.dataset.values[self.peptide_molecule][self.column]).groupby("id").mean().std()
        )
        flyability = estimate_flyability_upper_bound(
            dataset=self.dataset,
            column=self.column,
            mapping=self.mapping,
            protein_molecule=self.protein_molecule,
            peptide_molecule=self.peptide_molecule,
            remove_one=True,
            pbar=False,
        )
        self.flyability_dist = scipy.stats.beta.fit(flyability[~flyability.isna()])
        self.peptide_noise_level = self.dataset.get_column_flat(
            molecule=self.peptide_molecule, column=self.column
        ).quantile(noise_quantile)
        pep_abs = dataset.values[peptide_molecule][column]
        self.missingness = pep_abs.isna().sum() / pep_abs.shape[0]

    def create_mocked_dataset(
        self,
        molecule_set: Optional[MoleculeSet] = None,
        samples: int = None,
        simulate_missing: bool = True,
        random_seed: Optional[Union[int, np.random.Generator]] = None,
        noise_mnar_thresh_multiplier: float = 2,
        noise_mnar_thresh_std_multiplier: float = 1,
        mcar_frac: Optional[float] = None,
        print_parameters: bool = False,
    ) -> Dataset:
        random_seed = get_numpy_random_generator(random_seed)
        if samples is None:
            samples = len(self.dataset.samples_dict)
        if molecule_set is None:
            molecule_set = self.dataset.molecule_set
        sim_ds = simulate_protein_peptide_dataset(
            molecule_set=molecule_set,
            samples=samples,
            log_abundance_mu=self.log_abundance_mean,
            log_abundance_sigma=self.log_abundance_std,
            log_protein_error_sigma=0.3,
            simulate_flyability=True,
            flyability_alpha=self.flyability_dist[0],
            flyability_beta=self.flyability_dist[1],
            peptide_noise_sigma=self.peptide_noise_level,
            peptide_poisson_error=True,
            protein_column="abundance_gt",
            peptide_column="abundance",
            protein_molecule=self.protein_molecule,
            peptide_molecule=self.peptide_molecule,
            mapping=self.mapping,
            random_seed=random_seed,
            print_parameters=print_parameters
        )
        if simulate_missing:
            thresh_mu = self.peptide_noise_level * noise_mnar_thresh_multiplier
            thresh_sigma = self.peptide_noise_level * noise_mnar_thresh_std_multiplier
            if print_parameters:
                print(f"MNAR thres mu:{thresh_mu}, sigma:{thresh_sigma}")
            simulate_mnars_thresholding(
                dataset=sim_ds,
                thresh_mu=thresh_mu,
                thresh_sigma=thresh_sigma,
                molecule=self.peptide_molecule,
                column="abundance",
                result_column="abundance_missing",
                rng=random_seed,
                inplace=True,
                mask_column='is_mnar'
            )
            pep_abs = sim_ds.values[self.peptide_molecule]["abundance_missing"]
            mnar_missingness = pep_abs.isna().sum() / pep_abs.shape[0]
            if print_parameters:
                print(f"MNAR missingness: {mnar_missingness}")
            if mcar_frac is None:
                if mnar_missingness > self.missingness:
                    raise ValueError(
                        "MNAR missingness after MNAR threshold estimation already higher than overall missingness. "
                        "Missing value parameters cannot be estimated. Try specifying a different mnar_thresh multiplyer"
                    )
                mcar_frac = self.missingness - mnar_missingness
            if print_parameters:
                print(f"MCAR fraction: {mcar_frac}")
            simulate_mcars(
                dataset=sim_ds,
                amount=mcar_frac,
                rng=random_seed,
                molecule="peptide",
                column="abundance_missing",
                result_column="abundance_missing",
                inplace=True,
                mask_column='is_mcar',
                mask_only_non_missing = True,
            )
        return sim_ds
