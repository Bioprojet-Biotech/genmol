# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ligand preparation for gnina (protonation at physiological pH, pose preservation)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem

DEFAULT_DOCKING_PH = 7.4


def get_obabel_executable():
    return shutil.which('obabel')


def _run_obabel(argv, timeout=120):
    exe = get_obabel_executable()
    if not exe:
        return None
    try:
        return subprocess.run(
            [exe, *argv],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def protonate_sdf_file(input_path, output_path, ph=DEFAULT_DOCKING_PH, gen3d=False):
    """Protonate ligand SDF at pH; add 3D only when gen3d is True."""
    argv = [str(input_path), '-osdf', '-O', str(output_path), '-p', str(ph)]
    if gen3d:
        argv.append('--gen3d')
    else:
        argv.append('-h')
    result = _run_obabel(argv)
    if result is None or result.returncode != 0:
        return False
    return os.path.isfile(output_path) and os.path.getsize(output_path) > 0


def load_first_mol_from_sdf(path):
    try:
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        for mol in suppl:
            if mol is not None and mol.GetNumAtoms() > 0:
                return mol
    except Exception:
        return None
    return None


def write_mol_to_sdf(mol, path):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def _prepare_ligand_mol_rdkit_fallback(mol, embed_3d):
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m, catchErrors=True)
    except Exception:
        pass
    has_3d = m.GetNumConformers() > 0
    try:
        if has_3d:
            m = Chem.AddHs(m, addCoords=True)
        else:
            m = Chem.AddHs(m)
    except Exception:
        try:
            m = Chem.AddHs(m)
        except Exception:
            return None
    if embed_3d and m.GetNumConformers() == 0:
        try:
            if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
                return None
            try:
                AllChem.MMFFOptimizeMolecule(m)
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(m)
                except Exception:
                    pass
        except Exception:
            return None
    return m


def prepare_ligand_mol_for_docking(mol, ph=DEFAULT_DOCKING_PH, embed_3d=True):
    """
    Return a protonated RDKit mol with explicit Hs.

    When the input already has a 3D conformer, coordinates are preserved.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    has_3d = mol.GetNumConformers() > 0
    tf_in = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False)
    tf_in.close()
    tf_out = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False)
    tf_out.close()
    try:
        if not write_mol_to_sdf(mol, tf_in.name):
            return None
        if protonate_sdf_file(
            tf_in.name,
            tf_out.name,
            ph=ph,
            gen3d=embed_3d and not has_3d,
        ):
            prepared = load_first_mol_from_sdf(tf_out.name)
            if prepared is not None:
                return prepared
    finally:
        for path in (tf_in.name, tf_out.name):
            try:
                os.unlink(path)
            except OSError:
                pass

    return _prepare_ligand_mol_rdkit_fallback(mol, embed_3d=embed_3d and not has_3d)


def smiles_to_docking_mol_3d(smiles, ph=DEFAULT_DOCKING_PH):
    """Build a protonated 3D ligand mol from SMILES."""
    smi = (smiles or '').strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return prepare_ligand_mol_for_docking(mol, ph=ph, embed_3d=True)
