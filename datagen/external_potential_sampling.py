"""Module for sampling the external potential by placing small charges around the moleculse.

Different external potentials lead to different solutions of KS-DFT. Thus slightly perturbing the
external potential allows to sample different electron densities.

Attributes:
    RADI_METHOD (Dict): Available radi methods.

    PRUNE_METHOD (Dict): Available prune methods.

    SAMPLING_FUNCTIONS (Dict): Available sampling function.
"""
import copy
from typing import Callable, Generator, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pyscf import dft, gto

from mldft.utils.external_charges import DENOMINATOR_LIMIT, ExternalChargesMole

# TODO: do we want some consistency checks for the config prior to running the sampling?

RADI_METHOD = {
    "delley": dft.radi.delley,
    "becke": dft.radi.becke,
    "treutler": dft.radi.treutler,
    "mura-knowles": dft.radi.mura_knowles,
}

PRUNE_METHOD = {
    "none": None,
    "treutler": dft.treutler_prune,
    "sg1": dft.sg1_prune,
    "nwchem": dft.nwchem_prune,
}


def get_prune_method(prune_method_key: str) -> Callable:
    """Get the prune method from the prune method string.

    Args:
        prune_method (str): Prune method string.

    Returns:
        prune_method (Callable): Prune method.

    Raises:
        KeyError: If the prune method is not found in the PRUNE_METHOD dict.
    """
    try:
        prune_method = PRUNE_METHOD[prune_method_key.lower()]
    except KeyError:
        raise KeyError(
            f"Prune method {prune_method_key} not found choose from {list(PRUNE_METHOD.keys())}"
        )

    return prune_method


def get_radi_method(radi_method_key: str) -> Callable:
    """Get the radi method from the radi method string.

    Args:
        radi_method (str): Radi method string.

    Returns:
        radi_method (Callable): Radi method.

    Raises:
        KeyError: If the radi method is not found in the RADI_METHOD dict.
    """
    try:
        radi_method = RADI_METHOD[radi_method_key.lower()]
    except KeyError:
        raise KeyError(
            f"Radi method {radi_method_key} not found choose from {list(RADI_METHOD.keys())}"
        )

    return radi_method


def simple_sampling_per_atom(
    molecule: gto.Mole,
    charges_per_atom: int = 3,
    charge_value: int = 1000,
    grid_level=1,
    prune_method: str = "none",
    radi_method: str = "delley",
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple sampling of the external potential by placing charges randomly.

    Sampling is performed by placing charges at randomly selected grid (DFT integration) points
    for each atom based on the 'charges_per_atom' parameter in the config. The grid level used
    is determined by the 'grid_level' parameter. The charges are all set to the same value
    specified by the 'charge' parameter. The radi method used to generate the grid is determined
    by the 'radi_method' parameter. The prune method used to generate the grid is determined by
    the 'prune_method' parameter.

    The grid points are sampled uniformly from the grid points of the atom. This means that the
    charges are more likely to be placed in regions with higher electron density or more relevant
    for the system.

    Args:
        molecule: PySCF molecule object.
        charges_per_atom: Number of charges to place per atom.
        charge_value: Value of the charges. Will be divided by the denominator limit.
        grid_level: Level of the grid.
        prune_method: Method to prune the grid.
        radi_method: Method to generate the grid.

    Returns:
        charge_coords: Coordinates of the charges.
        charge_values: Values of the charges.
    """

    rng = np.random.default_rng(seed=42)

    partitioned_grids = generate_atomic_grids_for_charge_sampling(
        molecule,
        grid_level=grid_level,
        radi_method=radi_method,
        prune_method=prune_method,
    )
    charge_coords = []

    for i in range(molecule.natm):
        # the grid point coordinates for the i-th atom
        atom_grids_points = partitioned_grids[0][i]
        n_grid_points = atom_grids_points.shape[0]
        atom_grids_point_index = rng.integers(0, high=n_grid_points, size=charges_per_atom)
        atom_charge_coords = atom_grids_points[atom_grids_point_index, :]

        charge_coords.append(atom_charge_coords)

    charge_coords = np.vstack(charge_coords)

    charge_values = np.zeros(charge_coords.shape[0])
    charge_values[:] = charge_value / DENOMINATOR_LIMIT

    return charge_coords, charge_values


def advanced_sampling_per_atom(
    molecule: gto.Mole,
    max_number_charges_per_atom: int = 3,
    min_number_charges_per_atom: int = 0,
    min_charge_value: int = 1,
    max_charge_value: int = 1000,
    grid_level: int = 1,
    prune_method: str = "none",
    radi_method: str = "delley",
) -> Tuple[np.ndarray, np.ndarray]:
    """Advanced sampling of the external potential by placing charges randomly.

    This function allows for more advanced sampling of the external potential by placing a randomly
    selected number charges per atom with a randomly selected charge value. The number of charges
    per atom is determined by the 'min_charges_per_atom' and 'max_charges_per_atom' parameters in
    the config. The charge value is determined by the 'min_charge' and 'max_charge' parameters in
    the config. The actual charge is generated by randomly choosing integers in the range
    'min_charge' and 'max_charge' (inclusive) and dividing by the denominator limit (default:
    10000). The grid level used is determined by the 'grid_level' parameter. The radi method
    used to generate the grid is determined by the 'radi_method' parameter. The prune method used
    to generate the grid is determined by the 'prune_method' parameter.

    Args:
        molecule: PySCF molecule object.
        max_number_charges_per_atom: Maximum number of charges to place per atom.
        min_number_charges_per_atom: Minimum number of charges to place per atom.
        min_charge_value: Minimum value of the charges. Will be divided by the denominator limit.
        max_charge_value: Maximum value of the charges. Will be divided by the denominator limit.
        grid_level: Level of the grid.
        prune_method: Method to prune the grid.
        radi_method: Method to generate the grid.

    Returns:
        charge_coords: Coordinates of the charges.
        charge_values: Values of the charges.
    """
    rng = np.random.default_rng(seed=42)

    partitioned_grids = generate_atomic_grids_for_charge_sampling(
        molecule,
        grid_level=grid_level,
        radi_method=radi_method,
        prune_method=prune_method,
    )

    charge_coords = []

    for i in range(molecule.natm):
        charges_per_atom = rng.integers(
            min_number_charges_per_atom, high=max_number_charges_per_atom, endpoint=True
        )
        atom_grids_points = partitioned_grids[0][i]
        n_grid_points = atom_grids_points.shape[0]
        atom_grids_point_index = rng.integers(0, high=n_grid_points, size=charges_per_atom)
        atom_charge_coords = atom_grids_points[atom_grids_point_index, :]

        charge_coords.append(atom_charge_coords)

    charge_coords = np.vstack(charge_coords)

    charge_values = (
        rng.integers(
            min_charge_value,
            high=max_charge_value,
            size=charge_coords.shape[0],
            endpoint=True,
        )
        / DENOMINATOR_LIMIT
    )

    return charge_coords, charge_values


def sample_atomic_charge(
    molecule: gto.Mole,
    percentage_of_atoms_to_modify: int = 100,
    atomic_charge_modification_factor_min: float = 1.0,
    atomic_charge_modification_factor_max: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the atomic charges.

    The atomic charges are sampled by randomly selecting a number of atoms to modify based on the
    'percentage_of_atoms_to_modify' parameter. The atomic charge is then modified by multiplying
    the charge of the atom by the 'atomic_charge_modification_facotr' parameter and adding these
    charges to the same position in space as the respective atoms, thereby increasing the
    electron nulcear attraction.

    Args:
        molecule: PySCF molecule object.
        percentage_of_atoms_to_modify: Percentage of atoms for which the nuclear charge should be
            modified. Between 0 and 100.
        atomic_charge_modification_factor_min: Minimum factor to modify the atomic charge.
        atomic_charge_modification_factor_max: Maximum factor to modify the atomic charge.

    Returns:
        charge_coords: Coordinates of the charges.
        charge_values: Values of the charges.

    Note:
        Currently the atomic charges can only be increased and not decreased.
    """

    assert 0 <= percentage_of_atoms_to_modify <= 100, (
        "Percentage of atoms to modify must be between 0 and 100, but is"
        f"{percentage_of_atoms_to_modify}"
    )
    percentage_of_atoms_to_modify = percentage_of_atoms_to_modify / 100

    natoms_to_modify = int(np.ceil(molecule.natm * percentage_of_atoms_to_modify))

    rng = np.random.default_rng(seed=42)
    atom_indices = rng.choice(molecule.natm, size=natoms_to_modify, replace=False)

    atomic_charge_modification_facotr_min = atomic_charge_modification_factor_min - 1
    atomic_charge_modification_facotr_max = atomic_charge_modification_factor_max - 1
    assert (
        atomic_charge_modification_facotr_max >= 0 and atomic_charge_modification_facotr_min >= 0
    )

    atomic_charge_modification_facotr = rng.uniform(
        atomic_charge_modification_facotr_min,
        atomic_charge_modification_facotr_max,
        size=natoms_to_modify,
    )

    charge_coords = molecule.atom_coords()[atom_indices]
    charge_values = molecule.atom_charges()[atom_indices] * atomic_charge_modification_facotr
    charge_values = np.round(charge_values, decimals=4)

    return charge_coords, charge_values


def generate_atomic_grids_for_charge_sampling(
    molecule: gto.Mole,
    grid_level: int = 1,
    radi_method="delley",
    prune_method="none",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the grids for the charge sampling.

    Args:
        molecule (gto.Mole): PySCF molecule object.
        grid_level: Level of the grid.
        radi_method: Method to generate the grid.
        prune_method: Method to prune the grid.

    Returns:
        atomic_grids (np.ndarray): Atomic grids.
        partitioned_grids (np.ndarray): Partitioned grids.
    """
    # radi method that does not place grid points further away than 12 Bohr/ ~6 Angstrom for second
    # row elements charges placed far away might not have an effect on the system at all
    radi_method = get_radi_method(radi_method)
    prune_method = get_prune_method(prune_method)

    atomic_grids = dft.gen_grid.gen_atomic_grids(
        molecule, radi_method=radi_method, level=grid_level, prune=prune_method
    )
    # the grids for each atom translated to the respective atom position, tuple of arrays
    # (coords, weights)
    partitioned_grids = dft.gen_grid.gen_partition(molecule, atomic_grids, concat=False)

    return partitioned_grids


SAMPLING_FUNCTIONS = {
    "simple_per_atom": simple_sampling_per_atom,
    "advanced_per_atom": advanced_sampling_per_atom,
    "atomic_charge": sample_atomic_charge,
}


def get_sampling_function(sampling_function_key: str) -> Callable:
    """Get the sampling function from the sampling function string.

    Args:
        sampling_function_key (str): Sampling function string.

    Returns:
        sampling_function (Callable): Sampling function.

    Raises:
        KeyError: If the sampling function is not found in the SAMPLING_FUNCTIONS dict.
    """
    try:
        sampling_function = SAMPLING_FUNCTIONS[sampling_function_key.lower()]
    except KeyError:
        raise KeyError(
            f"Sampling function {sampling_function_key} not found choose from {list(SAMPLING_FUNCTIONS.keys())}"
        )

    return sampling_function


def external_potential_sampling(
    molecule: gto.Mole, cfg: DictConfig | dict
) -> Generator[gto.Mole, None, None]:
    """Sampling of external potential using the in the config specified sampling function.

    Args:
        molecule (gto.Mole): PySCF molecule object.
        cfg (DictConfig): Configuration object.

    Yields:
        molecule (gto.Mole): PySCF molecule object with external potential.
    """
    if isinstance(cfg, DictConfig):
        config_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        config_dict = cfg
    samples_per_structure = config_dict.pop("samples_per_structure", 1)
    sampling_key = config_dict["sampling_function"]["name"]
    sampling_function = get_sampling_function(sampling_key)
    sampling_config = copy.deepcopy(config_dict["sampling_function"])
    sampling_config.pop("name")

    for i in range(samples_per_structure):
        charge_coords, charge_values = sampling_function(molecule, **sampling_config)
        molecule_with_charges = ExternalChargesMole.from_mol(
            molecule, external_charges=charge_values, external_coords=charge_coords
        )

        yield molecule_with_charges


if __name__ == "__main__":
    cfg = OmegaConf.create(
        {
            "charge": 1.0,
            "charges_per_atom": 1,
            "grid_level": 1,
            "name": "simple_per_atom",
        }
    )

    molecule = gto.M(atom="F 0 0 0; H 0 0 1.1", basis="6-31G", spin=None)

    molecule = external_potential_sampling(molecule, cfg)
