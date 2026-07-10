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


def mol_contains_core(smiles, core_mol):
    """Return True if *smiles* is compatible with *core_mol* for constrained embedding.

    Uses the same criterion as constrained_embed_smiles: the molecule contains the
    core as a substructure, or the core contains the molecule.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        return False
    core = Chem.RemoveHs(Chem.Mol(core_mol))
    return mol.GetSubstructMatch(core) != () or core.GetSubstructMatch(mol) != ()


def constrained_embed_smiles(smiles, core_mol, random_seed=0):
    """
    Embed a SMILES molecule in 3D while matching the core substructure coordinates.

    Returns an RDKit mol with one conformer, or raises on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')

    core = Chem.RemoveHs(Chem.Mol(core_mol))
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


def subset_mol_preserve_conformer(mol, atom_indices, conf_id=0):
    """Build a sub-mol with selected atoms and 3D positions copied from *mol*."""
    if not atom_indices:
        raise ValueError('atom_indices is empty')
    atom_indices = sorted(set(atom_indices))
    if mol.GetNumConformers() == 0:
        raise ValueError('Molecule has no conformer')
    em = Chem.RWMol(Chem.Mol())
    old_to_new = {}
    for old_i in atom_indices:
        new_i = em.AddAtom(mol.GetAtomWithIdx(old_i))
        old_to_new[old_i] = new_i
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in old_to_new and b in old_to_new:
            em.AddBond(old_to_new[a], old_to_new[b], bond.GetBondType())
    out = em.GetMol()
    ref_conf = mol.GetConformer(conf_id)
    new_conf = Chem.Conformer(out.GetNumAtoms())
    for old_i in atom_indices:
        new_i = old_to_new[old_i]
        new_conf.SetAtomPosition(new_i, ref_conf.GetAtomPosition(old_i))
    out.RemoveAllConformers()
    out.AddConformer(new_conf, assignId=True)
    return out


def query_mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    return Chem.RemoveHs(mol)


def find_substructure_atom_indices(mol, smiles, match_index=0, use_chirality=True):
    """Return atom indices in *mol* that match the SMILES substructure."""
    query = query_mol_from_smiles(smiles)
    matches = mol.GetSubstructMatches(query, uniquify=True, useChirality=use_chirality)
    if not matches and use_chirality:
        matches = mol.GetSubstructMatches(query, uniquify=True, useChirality=False)
    if not matches:
        raise ValueError(f'SMILES substructure not found in molecule: {smiles}')
    if match_index < 0 or match_index >= len(matches):
        raise ValueError(
            f'match_index {match_index} out of range (found {len(matches)} match(es))'
        )
    return list(matches[match_index])


def load_mol_3d_record(path, record=0, conf_id=0):
    """Load one 3D structure record from SDF/MOL/PDB/PDBQT."""
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.sdf':
        supplier = Chem.SDMolSupplier(path, removeHs=False)
        mol = None
        for idx, candidate in enumerate(supplier):
            if idx == record:
                mol = candidate
                break
        if mol is None:
            raise ValueError(f'Record {record} not found in SDF: {path}')
    else:
        if record != 0:
            raise ValueError(f'record index is only supported for SDF input (got {ext})')
        mol = load_mol_3d(path)
    if mol is None:
        raise ValueError(f'Failed to read molecule from {path}')
    if mol.GetNumConformers() == 0:
        raise ValueError(f'No 3D conformer found in {path} record {record}')
    if conf_id != 0:
        if conf_id >= mol.GetNumConformers():
            raise ValueError(
                f'conf_id {conf_id} out of range (molecule has {mol.GetNumConformers()} conformer(s))'
            )
        conf = Chem.Conformer(mol.GetConformer(conf_id))
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
    return mol


def extract_substructure_3d(mol, smiles, match_index=0, conf_id=0, use_chirality=True):
    """Extract atoms matching *smiles* from *mol*, preserving 3D coordinates."""
    query = query_mol_from_smiles(smiles)
    atom_indices = find_substructure_atom_indices(
        mol, smiles, match_index=match_index, use_chirality=use_chirality,
    )
    subset = subset_mol_preserve_conformer(mol, atom_indices, conf_id=conf_id)
    try:
        subset = Chem.AssignBondOrdersFromTemplate(query, subset)
    except Exception:
        pass
    return subset


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
    try:
        writer.SetKekulize(False)
    except AttributeError:
        pass
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
