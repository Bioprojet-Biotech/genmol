# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .ligand_prep import DEFAULT_DOCKING_PH, prepare_ligand_mol_for_docking, smiles_to_docking_mol_3d


def load_mol_3d(path):
    """Load a molecule with at least one 3D conformer from SDF/MOL/PDB/PDBQT."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'3D structure file not found: {path}')

    ext = os.path.splitext(path)[1].lower()
    mol = None
    if ext == '.sdf':
        supplier = Chem.SDMolSupplier(path, removeHs=False)
        mol = supplier[0] if supplier else None
    elif ext == '.mol':
        mol = Chem.MolFromMolFile(path, removeHs=False)
    elif ext == '.pdb':
        mol = Chem.MolFromPDBFile(path, removeHs=False)
    elif ext == '.pdbqt':
        with tempfile.NamedTemporaryFile(suffix='.mol', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.check_output(
                ['obabel', path, '-O', tmp_path],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            mol = Chem.MolFromMolFile(tmp_path, removeHs=False)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        raise ValueError(
            f'Unsupported core 3D format "{ext}" for {path}; '
            'use .sdf, .mol, .pdb, or .pdbqt'
        )

    if mol is None:
        raise ValueError(f'Failed to read molecule from {path}')
    if mol.GetNumConformers() == 0:
        raise ValueError(f'No 3D conformer found in {path}')
    return mol


def smiles_from_mol(mol):
    """Return canonical heavy-atom SMILES for a molecule."""
    mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    if not smiles:
        raise ValueError('Failed to derive SMILES from 3D structure')
    return smiles


def smiles_from_mol_3d(path):
    """Load a 3D structure file and return its canonical SMILES."""
    return smiles_from_mol(load_mol_3d(path))


def constrained_embed_smiles(smiles, core_mol, random_seed=0):
    """
    Embed a SMILES molecule in 3D while matching the core substructure coordinates.

    Returns an RDKit mol with one conformer, or raises on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')

    core = Chem.Mol(core_mol)
    if mol.GetSubstructMatch(core) == () and core.GetSubstructMatch(mol) == ():
        raise ValueError(
            'SMILES does not contain the core substructure used for constrained embedding'
        )

    mol = Chem.AddHs(mol)
    embedded = AllChem.ConstrainedEmbed(
        mol,
        core,
        useTethers=True,
        randomseed=random_seed,
    )
    return Chem.RemoveHs(embedded)


def write_mol_file(mol, path):
    """Write an RDKit mol with 3D coordinates to a .mol file."""
    if mol.GetNumConformers() == 0:
        raise ValueError('Molecule has no conformer to write')
    Chem.MolToMolFile(mol, path, kekulize=False)


def write_mol_sdf(mol, path):
    """Write an RDKit mol with 3D coordinates to an SDF file."""
    if mol.GetNumConformers() == 0:
        raise ValueError('Molecule has no conformer to write')
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def heavy_atom_coords_array(mol):
    conf = mol.GetConformer()
    pts = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        pts.append([p.x, p.y, p.z])
    return np.asarray(pts, dtype=float)


def translate_mol_centroid_to_point(mol, target_xyz):
    """Shift mol so the heavy-atom centroid sits at target_xyz."""
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    coords = heavy_atom_coords_array(out)
    if coords.size == 0:
        return out
    centroid = coords.mean(axis=0)
    target = np.asarray(target_xyz, dtype=float)
    delta = target - centroid
    for i in range(out.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(
            i,
            Chem.rdGeometry.Point3D(
                float(p.x + delta[0]),
                float(p.y + delta[1]),
                float(p.z + delta[2]),
            ),
        )
    return out


def build_initial_mol_for_gnina(smiles, core_mol=None, box_center=None, random_seed=0, ph=DEFAULT_DOCKING_PH):
    """
    Build a protonated 3D ligand for gnina local minimize.

    With core_mol: ConstrainedEmbed then protonate preserving pose.
    Without core_mol: embed from SMILES, translate centroid to box_center, protonate.
    """
    if core_mol is not None:
        raw = constrained_embed_smiles(smiles, core_mol, random_seed=random_seed)
    else:
        raw = smiles_to_docking_mol_3d(smiles, ph=ph)
        if raw is None:
            raise ValueError(f'Could not build 3D structure from SMILES: {smiles}')
        if box_center is not None:
            raw = translate_mol_centroid_to_point(raw, box_center)

    prepared = prepare_ligand_mol_for_docking(raw, ph=ph, embed_3d=False)
    if prepared is None:
        raise ValueError(f'Failed to protonate ligand for gnina: {smiles}')
    return prepared
