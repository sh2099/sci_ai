"""Compute labels for molecules in the dataset and save them as zarr.zip files.

Can be run with the `mldft_labelgen` command.
"""

import multiprocessing
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Callable, Tuple

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.datagen.methods.label_generation import str_to_calculation_fct
from mldft.datagen.methods.save_labels_in_zarr_file import save_density_fitted_data
from mldft.ofdft.run_ofdft import parse_run_path, run_to_checkpoint_path
from mldft.utils import extras
from mldft.utils.external_charges import load_external_charges_to_mol_from_chk
from mldft.utils.molecules import build_molecule_np, construct_aux_mol, load_scf
from mldft.utils.multiprocess import (
    configure_max_memory_per_process,
    configure_processes_and_threads,
)
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula


def get_id_and_sample_id_from_chk_file(file: Path) -> Tuple[int, int | None]:
    """Get the molecule id and sample id from the .chk file.

    Args:
        file: The .chk file.

    Returns:
        Tuple of molecule id and sample id.
    """
    split_filename = file.stem.split("_")[-1].split(".")
    if len(split_filename) == 2:
        molecule_id, sample_id = int(split_filename[0]), int(split_filename[1])
    else:
        molecule_id = int(split_filename[0])
        sample_id = None
    return molecule_id, sample_id


def get_zarr_file_path(
    label_dir: Path,
    molecule_id: int,
    sample_id: int | None = None,
) -> Path:
    """Get the path to the zarr file for the molecule.

    Args:
        label_dir: The directory containing the label files.
        molecule_id: The molecule id.
        sample_id: The sample id.

    Returns:
        The path to the zarr file.
    """
    if sample_id is not None:
        label_path = label_dir / f"{molecule_id:07}.{sample_id:07}.zarr.zip"
    else:
        label_path = label_dir / f"{molecule_id:07}.zarr.zip"
    return label_path


def labelgen_and_save_single_process(
    dataset: DataGenDataset,
    calculation_fct: Callable[..., dict],
    chk_files: Sequence[Path],
    orbital_basis: str,
    kohn_sham_settings: DictConfig,
    of_basis_set: str,
):
    """Label generation and saving for all molecules in a single process.

    Args:
        dataset: The dataset object.
        calculation_fct: The calculation function to use.
        chk_files: List of paths to the .chk files.
        orbital_basis: The orbital basis to use.
        kohn_sham_settings: The kohn-sham settings.
        of_basis_set: The orbital free basis set.
    """
    for file in tqdm(chk_files, position=0, dynamic_ncols=True, total=len(chk_files)):
        molecule_id, sample_id = get_id_and_sample_id_from_chk_file(file)
        label_path = get_zarr_file_path(
            dataset.label_dir,
            molecule_id,
            sample_id=sample_id,
        )
        results, initialization, data_of_iteration = load_scf(file)
        mol_orbital_basis = dataset.load_molecule(molecule_id, orbital_basis)
        mol_density_basis = construct_aux_mol(
            mol_orbital_basis, aux_basis_name=of_basis_set, unit="Bohr"
        )
        logger.info(
            f"Computing molecule {molecule_id} {mole_to_sum_formula(mol_orbital_basis, True)} with {len(mol_orbital_basis.atom_charges())} atoms."
        )

        # add external charges to mol_orbital_basis and mol_density_basis if they are encountered
        # in the chk file else it leaves them unchanged
        mol_orbital_basis = load_external_charges_to_mol_from_chk(file, mol=mol_orbital_basis)
        mol_density_basis = load_external_charges_to_mol_from_chk(file, mol=mol_density_basis)
        data = calculation_fct(
            results=results,
            initialization=initialization,
            data_of_iteration=data_of_iteration,
            mol_orbital_basis=mol_orbital_basis,
            mol_density_basis=mol_density_basis,
        )
        save_density_fitted_data(
            label_path,
            mol_density_basis,
            kohn_sham_basis=kohn_sham_settings.basis,
            path_to_basis_info=(file.parent / "basis_info.npz").as_posix(),
            molecular_data=data,
            mol_id=file.stem,
        )


def labelgen_and_save_parallel_iteration(args: tuple):
    """Label generation and saving for a single molecule in a parallel process.

    Args:
        args: Tuple of file, dataset, calculation_fct, orbital_basis, kohn_sham_basis, of_basis_set.
    """
    (
        file,
        dataset,
        calculation_fct,
        orbital_basis,
        kohn_sham_basis,
        of_basis_set,
    ) = args
    molecule_id, sample_id = get_id_and_sample_id_from_chk_file(file)
    charges, positions = dataset.load_charges_and_positions(molecule_id)
    mol_orbital_basis = build_molecule_np(charges, positions, basis=orbital_basis, unit="Angstrom")
    mol_density_basis = construct_aux_mol(
        mol_orbital_basis, aux_basis_name=of_basis_set, unit="Bohr"
    )
    logger.info(
        f"Computing molecule {molecule_id} {mole_to_sum_formula(mol_orbital_basis, True)} with {len(mol_orbital_basis.atom_charges())} atoms."
    )

    # add external charges to mol_orbital_basis and mol_density_basis if they are encountered
    # in the chk file else it leaves them unchanged
    mol_orbital_basis = load_external_charges_to_mol_from_chk(file, mol=mol_orbital_basis)
    mol_density_basis = load_external_charges_to_mol_from_chk(file, mol=mol_density_basis)

    label_path = get_zarr_file_path(
        dataset.label_dir,
        molecule_id,
        sample_id=sample_id,
    )
    results, initialization, data_of_iteration = load_scf(file)
    data = calculation_fct(
        results=results,
        initialization=initialization,
        data_of_iteration=data_of_iteration,
        mol_orbital_basis=mol_orbital_basis,
        mol_density_basis=mol_density_basis,
    )
    save_density_fitted_data(
        label_path,
        mol_density_basis,
        kohn_sham_basis=kohn_sham_basis,
        path_to_basis_info=(file.parent / "basis_info.npz").as_posix(),
        molecular_data=data,
        mol_id=file.stem,
    )


def labelgen_and_save_parallel(
    num_processes: int,
    dataset: DataGenDataset,
    calculation_fct: Callable,
    chk_files: Sequence[Path],
    orbital_basis: str,
    kohn_sham_settings: DictConfig,
    of_basis_set: str,
):
    """Set up the parallel processes and run the label generation and saving.

    Args:
        num_processes: Number of processes to use.
        dataset: The dataset object.
        calculation_fct: The calculation function to use.
        chk_files: List of paths to the .chk files.
        orbital_basis: The orbital basis to use.
        kohn_sham_settings: The kohn-sham settings.
        of_basis_set: The orbital free basis set.
    """
    args_list = [
        (
            file,
            dataset,
            calculation_fct,
            orbital_basis,
            kohn_sham_settings.basis,
            of_basis_set,
        )
        for file in chk_files
    ]
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Using imap_unordered instead of starmap_async for the progress bar
        imap = pool.imap_unordered(labelgen_and_save_parallel_iteration, args_list)
        for _ in tqdm(imap, total=len(chk_files), position=0, smoothing=0, dynamic_ncols=True):
            pass


def save_dataset_info(cfg: DictConfig, label_dir: Path):
    """Saves a few keywords from the config to a yaml file."""
    keys_to_save = ("kohn_sham", "density_fitting_method", "of_basis_set", "dataset")
    dataset_info = {}
    for key in keys_to_save:
        dataset_info[key] = cfg.get(key)
    OmegaConf.save(config=dataset_info, f=label_dir.joinpath("dataset_info.yaml"))


def run_label_generation(cfg: DictConfig):
    """Run the label generation and save as zarr.zip file for all molecules in the dataset.

    Args:
        cfg: The hydra config.
    """
    # Setup logger to log to hydra logdir and to the progress bar
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )
    num_processes = cfg.get("num_processes")
    num_threads_per_process = cfg.get("num_threads_per_process")
    # Load Dataset
    dataset_settings = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_settings["num_processes"] = num_processes
    dataset = hydra.utils.instantiate(dataset_settings)
    kohn_sham_settings = OmegaConf.load(dataset.kohn_sham_data_dir / "config.yaml")
    if not kohn_sham_settings == cfg.kohn_sham:
        raise ValueError(f"Settings do not match ({kohn_sham_settings})({cfg.kohn_sham}).")
    orbital_basis = kohn_sham_settings.basis
    assert dataset.kohn_sham_data_dir.exists()
    # Copy the config.yaml into the directory above the labels directory so that the datagen parameters are saved
    save_dataset_info(cfg, dataset.label_dir / "..")
    density_fitting_method = cfg.get("density_fitting_method")
    of_basis_set = cfg.get("of_basis_set")
    ids_this_run = dataset.get_ids_todo_labelgen(
        start_idx=cfg.start_idx, max_num_molecules=cfg.n_molecules
    )
    logger.info(
        f"Configured {cfg.n_molecules} starting from idx {cfg.start_idx}, computing {len(ids_this_run)} undone molecules."
    )
    chk_files = dataset.get_all_chk_files_from_ids(ids_this_run)

    calculation_fct_name = cfg.label_src.calculation_fct
    calculation_fct = str_to_calculation_fct[calculation_fct_name]
    calculation_fct = partial(calculation_fct, density_fitting_method=density_fitting_method)
    if calculation_fct_name == "active_learning":
        run_path = parse_run_path(cfg.label_src.run_path)
        iterations_per_mol = cfg.label_src.iterations_per_mol
        max_scale = cfg.label_src.max_scale

        # check if the dataset is compatible with the model
        model_config = OmegaConf.load(run_path / ".hydra/config.yaml")
        model_dataset_name = model_config.data.dataset_name
        dataset_compatible = model_dataset_name.split("_")[0] == dataset.name.split("_")[0]
        if not dataset_compatible:
            logger.warning(f"Dataset mismatch: {dataset.name} vs {model_dataset_name}")

        model_config.data.transforms.use_cached_data = False
        model_config.data.transforms.add_transformation_matrix = True
        master_transform = hydra.utils.instantiate(model_config.data.transforms)
        basis_info = hydra.utils.instantiate(model_config.data.basis_info)
        checkpoint_path = run_to_checkpoint_path(run_path)
        torch.set_default_dtype(torch.float64)
        calculation_fct = partial(
            calculation_fct,
            checkpoint_path=checkpoint_path,
            master_transform=master_transform,
            basis_info=basis_info,
            iterations_per_mol=iterations_per_mol,
            max_scale=max_scale,
        )

    num_processes = min(num_processes, len(chk_files))
    if num_processes <= 1:
        if num_threads_per_process == 1:
            logger.warning("Running in single process mode using 1 thread, this may take a while")
        else:
            logger.info(f"Running in single process mode using {num_threads_per_process} threads.")
        labelgen_and_save_single_process(
            dataset,
            calculation_fct,
            chk_files,
            orbital_basis,
            kohn_sham_settings,
            of_basis_set,
        )
    elif num_processes > 1:
        logger.info(
            f"Using {num_processes} processes with {num_threads_per_process} threads each."
        )
        labelgen_and_save_parallel(
            num_processes,
            dataset,
            calculation_fct,
            chk_files,
            orbital_basis,
            kohn_sham_settings,
            of_basis_set,
        )
    logger.info("Label generation and saving done.")


@hydra.main(version_base="1.3", config_path="../../configs/datagen", config_name="config.yaml")
def main(cfg: DictConfig):
    """Hydra entry point for the Kohn-Sham computation on the whole dataset. Sets up the hydra
    specific logging and then calls the run_label_generation() function.

    Args:
        cfg: The hydra config.
    """
    logger.add(cfg.log_file, rotation="10 MB", enqueue=True, backtrace=True, diagnose=True)

    # Setup number of concurrent processes and threads
    cfg.num_processes, cfg.num_threads_per_process = configure_processes_and_threads(
        cfg.get("num_processes"), cfg.get("num_threads_per_process")
    )
    cfg.max_memory_per_process = configure_max_memory_per_process(
        cfg.get("max_memory_per_process")
    )

    extras(cfg)
    run_label_generation(cfg)


if __name__ == "__main__":
    main()
