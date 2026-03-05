# RNA Trajectory Dataset for temporal conformation prediction.
#
# Given GROMACS MD trajectory PDB files, this dataset produces
# (input_feature_dict, label_dict) pairs where:
#   - input_feature_dict contains standard Protenix features + delta_t
#   - label_dict contains t2 atom coordinates as ground truth
#
# Usage:
#   dataset = RNATrajectoryDataset(
#       data_dir="training_dataset/",
#       cache_dir="training_dataset_cache/",
#   )
#   sample = dataset[0]
#   # sample["input_feature_dict"]["delta_t"]  -> tensor([500.0])
#   # sample["label_dict"]["coordinate"]       -> [N_atom, 3]

import os
import re
import pickle
import gzip
import random
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io import load_structure
from torch.utils.data import Dataset

from protenix.data.core.featurizer import Featurizer
from protenix.data.core.parser import AddAtomArrayAnnot
from protenix.data.inference.json_parser import add_entity_atom_array
from protenix.data.inference.json_to_feature import SampleDictToFeatures
from protenix.data.tokenizer import AtomArrayTokenizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_time_ps(filename: str) -> int:
    """Extract time in ps from filename like '..._decoys_500ps.pdb'."""
    m = re.search(r"_(\d+)ps\.pdb$", filename)
    if m is None:
        raise ValueError(f"Cannot extract time from filename: {filename}")
    return int(m.group(1))


def _extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extract single-letter RNA sequence from a PDB file."""
    atoms = load_structure(pdb_path)
    seen = set()
    seq = []
    for res_id, res_name in zip(atoms.res_id, atoms.res_name):
        if res_id not in seen:
            seen.add(res_id)
            seq.append(res_name)
    return "".join(seq)


def _build_reference_features(sequence: str) -> dict[str, Any]:
    """
    Build a Protenix-compatible feature dict from an RNA sequence string.

    Uses the inference pipeline (SampleDictToFeatures) to construct a
    fully annotated atom_array + token_array + feature_dict from just
    the RNA sequence. This gives us CCD-based atom ordering, ref_pos,
    and all other annotations that Featurizer needs.

    Returns:
        dict with keys: "feature_dict", "atom_array", "token_array",
                        "atom_name_list", "res_id_list"
    """
    # Build inference-style input JSON
    sample_dict = {
        "sequences": [
            {"rnaSequence": {"sequence": sequence, "count": 1}}
        ],
        "name": "rna_traj",
    }

    converter = SampleDictToFeatures(sample_dict)
    feature_dict, atom_array, token_array = converter.get_feature_dict()

    # Add dummy MSA/template features (will be replaced later if needed)
    feature_dict = make_dummy_feature(
        features_dict=feature_dict,
        dummy_feats=["msa", "template"],
    )
    feature_dict = data_type_transform(feat_or_label_dict=feature_dict)

    # Store atom identity for PDB→Protenix coordinate mapping
    atom_name_list = list(atom_array.atom_name)
    res_id_list = list(atom_array.res_id)

    return {
        "feature_dict": feature_dict,
        "atom_array": atom_array,
        "token_array": token_array,
        "atom_name_list": atom_name_list,
        "res_id_list": res_id_list,
    }


def _map_pdb_coords_to_protenix(
    pdb_path: str,
    protenix_atom_names: list[str],
    protenix_res_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load coordinates from a PDB file and map them to Protenix atom ordering.

    Protenix uses CCD-based atom layout (no hydrogens, 1-based res_ids).
    The PDB files may have different res_id numbering (e.g. 201-based)
    and include hydrogens. We match by (residue_order, atom_name), where
    residue_order is the sequential index of each unique res_id.

    Args:
        pdb_path: path to PDB file
        protenix_atom_names: atom names in Protenix order
        protenix_res_ids: residue IDs in Protenix order (1-based)

    Returns:
        coords: [N_atom, 3] coordinates in Protenix order
        mask: [N_atom] binary mask (1 if atom found in PDB, 0 otherwise)
    """
    atoms = load_structure(pdb_path)

    # Map PDB res_ids to sequential order (1-based to match Protenix)
    pdb_unique_res = []
    seen = set()
    for r in atoms.res_id:
        if r not in seen:
            seen.add(r)
            pdb_unique_res.append(r)
    pdb_resid_to_order = {r: i + 1 for i, r in enumerate(pdb_unique_res)}

    # Build lookup: (residue_order, atom_name) -> coord
    pdb_lookup = {}
    for i in range(len(atoms)):
        order = pdb_resid_to_order[atoms.res_id[i]]
        key = (order, atoms.atom_name[i])
        pdb_lookup[key] = atoms.coord[i]

    # GROMACS PDB uses old-style atom names for phosphate oxygens
    _ATOM_NAME_ALIASES = {
        "OP1": "O1P",
        "OP2": "O2P",
    }

    n_atoms = len(protenix_atom_names)
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    mask = np.zeros(n_atoms, dtype=np.float32)

    for i, (res_id, atom_name) in enumerate(
        zip(protenix_res_ids, protenix_atom_names)
    ):
        key = (int(res_id), atom_name)
        if key in pdb_lookup:
            coords[i] = pdb_lookup[key]
            mask[i] = 1.0
        else:
            # Try alias
            alias = _ATOM_NAME_ALIASES.get(atom_name)
            if alias:
                alt_key = (int(res_id), alias)
                if alt_key in pdb_lookup:
                    coords[i] = pdb_lookup[alt_key]
                    mask[i] = 1.0

    return coords, mask


class RNATrajectoryDataset(Dataset):
    """
    Dataset for RNA MD trajectory temporal prediction.

    Each sample is a (t1, t2) pair from the same RNA trajectory:
    - input_feature_dict: standard Protenix features (from RNA sequence)
                          + delta_t (time difference in ps)
    - label_dict: t2 atom coordinates as ground truth

    Args:
        data_dir: directory containing RNA trajectory subdirectories.
            Each subdirectory has PDB files named {ID}_..._decoys_{time}ps.pdb
        cache_dir: directory to cache preprocessed reference features.
            If None, uses data_dir + "_cache"
        min_delta_t: minimum time difference in ps (default: 100)
        max_delta_t: maximum time difference in ps (default: 99900)
        ref_pos_augment: whether to augment ref_pos with random rotation (default: True)
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        min_delta_t: float = 100.0,
        max_delta_t: float = 99900.0,
        ref_pos_augment: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir or (data_dir.rstrip("/") + "_cache")
        self.min_delta_t = min_delta_t
        self.max_delta_t = max_delta_t
        self.ref_pos_augment = ref_pos_augment

        # Discover RNA trajectories
        self.trajectories = self._discover_trajectories()
        logger.info(
            f"RNATrajectoryDataset: found {len(self.trajectories)} RNA trajectories "
            f"in {data_dir}"
        )

        # Build index: list of (traj_idx, t1_ps, t2_ps) triples
        # For training, we pre-generate all valid pairs per trajectory
        self.samples = self._build_sample_index()
        logger.info(f"RNATrajectoryDataset: {len(self.samples)} total samples")

        # Cache for preprocessed reference features (per RNA)
        self._ref_cache: dict[int, dict] = {}

    def _discover_trajectories(self) -> list[dict[str, Any]]:
        """Scan data_dir for RNA trajectory subdirectories."""
        trajectories = []
        for entry in sorted(os.listdir(self.data_dir)):
            subdir = os.path.join(self.data_dir, entry)
            if not os.path.isdir(subdir):
                continue
            # Find all PDB files and extract time points
            pdb_files = sorted(
                [f for f in os.listdir(subdir) if f.endswith(".pdb")]
            )
            if len(pdb_files) == 0:
                continue
            time_points = {}
            for f in pdb_files:
                try:
                    t = _extract_time_ps(f)
                    time_points[t] = os.path.join(subdir, f)
                except ValueError:
                    continue
            if len(time_points) < 2:
                continue
            trajectories.append(
                {
                    "name": entry,
                    "dir": subdir,
                    "time_points": time_points,
                    "sorted_times": sorted(time_points.keys()),
                }
            )
        return trajectories

    def _build_sample_index(self) -> list[tuple[int, int, int]]:
        """
        Build training samples: for each trajectory, enumerate all valid
        (t1, t2) pairs where min_delta_t <= |t2 - t1| <= max_delta_t.

        For efficiency, we don't enumerate ALL O(N^2) pairs.
        Instead, for each t1, we randomly sample a few t2 values.
        This is called once; actual t2 sampling also happens in __getitem__.
        """
        samples = []
        for traj_idx, traj in enumerate(self.trajectories):
            times = traj["sorted_times"]
            # Each time point is a potential t1; t2 is sampled at runtime
            for t1 in times:
                samples.append((traj_idx, t1))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_ref_features(self, traj_idx: int) -> dict[str, Any]:
        """
        Get preprocessed reference features for a trajectory.
        Uses cache (in-memory and on-disk).
        """
        if traj_idx in self._ref_cache:
            return self._ref_cache[traj_idx]

        traj = self.trajectories[traj_idx]
        cache_path = os.path.join(self.cache_dir, f"{traj['name']}_ref.pkl.gz")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached features for {traj['name']}")
            with gzip.open(cache_path, "rb") as f:
                ref_data = pickle.load(f)
        else:
            logger.info(f"Preprocessing {traj['name']}...")
            # Extract sequence from any PDB file (they all share the same topology)
            first_pdb = traj["time_points"][traj["sorted_times"][0]]
            sequence = _extract_sequence_from_pdb(first_pdb)
            ref_data = _build_reference_features(sequence)

            # Save cache
            os.makedirs(self.cache_dir, exist_ok=True)
            with gzip.open(cache_path, "wb") as f:
                pickle.dump(ref_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Cached features for {traj['name']} -> {cache_path}")

        self._ref_cache[traj_idx] = ref_data
        return ref_data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Returns a training sample dict:
        {
            "input_feature_dict": {
                ...standard Protenix features...,
                "delta_t": tensor([delta_t_ps]),
            },
            "label_dict": {
                "coordinate": tensor([N_atom, 3]),      # t2 coords
                "coordinate_mask": tensor([N_atom]),     # valid atom mask
            },
            "label_full_dict": {},
            "basic": {
                "pdb_id": str,
                "N_token": tensor,
                "N_atom": tensor,
                ...
            },
        }
        """
        for attempt in range(10):
            try:
                return self._process_one(idx)
            except Exception as e:
                logger.warning(
                    f"[RNATrajectoryDataset] Error at idx {idx} "
                    f"(attempt {attempt+1}): {e}\n{traceback.format_exc()}"
                )
                idx = random.randrange(len(self.samples))
        raise RuntimeError(f"Failed to load sample after 10 attempts")

    def _process_one(self, idx: int) -> dict[str, Any]:
        traj_idx, t1_ps = self.samples[idx]
        traj = self.trajectories[traj_idx]

        # Sample t2 randomly (different from t1, within delta_t range)
        valid_t2 = [
            t for t in traj["sorted_times"]
            if t != t1_ps
            and self.min_delta_t <= abs(t - t1_ps) <= self.max_delta_t
        ]
        if len(valid_t2) == 0:
            # Fallback: use any other time
            valid_t2 = [t for t in traj["sorted_times"] if t != t1_ps]
        t2_ps = random.choice(valid_t2)
        delta_t = float(abs(t2_ps - t1_ps))

        # Get preprocessed reference features (sequence-level, shared across all frames)
        ref_data = self._get_ref_features(traj_idx)
        atom_names = ref_data["atom_name_list"]
        res_ids = ref_data["res_id_list"]

        # Load t2 coordinates (label)
        t2_path = traj["time_points"][t2_ps]
        t2_coords, t2_mask = _map_pdb_coords_to_protenix(
            t2_path, atom_names, res_ids
        )

        # Deep copy feature_dict to avoid mutating cache
        import copy
        feature_dict = copy.deepcopy(ref_data["feature_dict"])

        # Add delta_t
        feature_dict["delta_t"] = torch.tensor([delta_t], dtype=torch.float32)

        # Add is_distillation (required by Protenix forward)
        if "is_distillation" not in feature_dict:
            feature_dict["is_distillation"] = torch.tensor([False])

        # Build label_dict
        label_dict = {
            "coordinate": torch.tensor(t2_coords, dtype=torch.float32),
            "coordinate_mask": torch.tensor(t2_mask, dtype=torch.long),
        }

        # Basic info
        n_atom = len(atom_names)
        n_token = int(feature_dict["token_index"].shape[0])
        basic_info = {
            "pdb_id": traj["name"],
            "N_token": torch.tensor([n_token]),
            "N_atom": torch.tensor([n_atom]),
            "N_asym": torch.tensor([1]),
            "t1_ps": t1_ps,
            "t2_ps": t2_ps,
            "delta_t_ps": delta_t,
            "N_msa": feature_dict["msa"].shape[0:1]
            if "msa" in feature_dict
            else torch.tensor([0]),
            "N_msa_prot_pair": torch.tensor([0]),
            "N_msa_prot_unpair": torch.tensor([0]),
            "N_msa_rna_pair": torch.tensor([0]),
            "N_msa_rna_unpair": torch.tensor([0]),
        }

        return {
            "input_feature_dict": feature_dict,
            "label_dict": label_dict,
            "label_full_dict": {},
            "basic": basic_info,
        }
