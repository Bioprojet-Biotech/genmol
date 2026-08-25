# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receptor PDB preparation for gnina (co-crystal extraction, apo stripping)."""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


PDB_SOLVENT_ION_RES = frozenset({
    'HOH', 'WAT', 'H2O', 'DOD', 'SO4', 'PO4', 'EDO', 'ACT', 'GOL', 'FMT', 'PEG', 'MPD',
    'DMS', 'CL', 'NA', 'K', 'MG', 'CA', 'BR', 'IOD', 'ZN', 'FE', 'MN', 'CO', 'NI', 'CD',
})
_MIN_LIGAND_HEAVY_ATOMS = 4


def _subset_mol_preserve_conformer(mol, atom_indices, conf_id=0):
    atom_indices = sorted(set(atom_indices))
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


@dataclass(frozen=True)
class LigandResidueKey:
    chain_id: str
    residue_number: int
    residue_name: str
    insertion_code: str = ' '


def stereo_insensitive_smiles_key(smiles):
    smi = (smiles or '').strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=False, kekuleSmiles=False, canonical=True)
    except Exception:
        return None


def mol_to_stereo_insensitive_key(mol):
    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None
    return stereo_insensitive_smiles_key(smi)


def mol_matches_smiles_stereo_insensitive(mol, smiles):
    mol_key = mol_to_stereo_insensitive_key(mol)
    smi_key = stereo_insensitive_smiles_key(smiles)
    return bool(mol_key and smi_key and mol_key == smi_key)


def assign_bond_orders_from_smiles(mol, smiles):
    """Assign bond orders from a SMILES template while preserving 3D coordinates."""
    smi = (smiles or '').strip()
    if mol is None or mol.GetNumConformers() == 0 or not smi:
        return None
    template = Chem.MolFromSmiles(smi)
    if template is None:
        return None
    try:
        had_hs = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
        heavy = Chem.RemoveHs(mol)
        ordered = AllChem.AssignBondOrdersFromTemplate(template, heavy)
        Chem.SanitizeMol(ordered)
        if had_hs:
            ordered = Chem.AddHs(ordered, addCoords=True)
        return ordered
    except Exception:
        return None


def ligand_residue_keys_from_mol(mol):
    keys = set()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        icode = info.GetInsertionCode() or ' '
        keys.add(LigandResidueKey(
            chain_id=info.GetChainId() or ' ',
            residue_number=int(info.GetResidueNumber()),
            residue_name=info.GetResidueName().strip(),
            insertion_code=icode if icode else ' ',
        ))
    return frozenset(keys)


def _heavy_atom_count(mol):
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _pdb_line_residue_key(line):
    if len(line) < 27:
        return None
    record = line[0:6].strip()
    if record not in ('ATOM', 'HETATM'):
        return None
    resname = line[17:20].strip()
    chain_id = line[21:22] if len(line) > 21 else ' '
    seq_field = line[22:26].strip()
    if not seq_field:
        return None
    try:
        resnum = int(seq_field)
    except ValueError:
        return None
    icode = line[26:27] if len(line) > 26 else ' '
    return LigandResidueKey(
        chain_id=chain_id or ' ',
        residue_number=resnum,
        residue_name=resname,
        insertion_code=icode if icode else ' ',
    )


def strip_ligand_from_pdb(pdb_path, ligand_residue_keys, output_path):
    if not ligand_residue_keys:
        src = Path(pdb_path).expanduser()
        out = Path(output_path)
        out.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
        return out
    src = Path(pdb_path).expanduser()
    kept = []
    for line in src.read_text(encoding='utf-8').splitlines(keepends=True):
        key = _pdb_line_residue_key(line)
        if key is not None and key in ligand_residue_keys:
            continue
        kept.append(line)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(kept), encoding='utf-8')
    return out


def normalize_pdb_for_gnina(pdb_path, output_path):
    """Write a gnina-friendly single-model PDB copy.

    Discovery Studio exports often wrap atoms in MODEL/ENDMDL; gnina/smina can
    refuse those. Always staging a local copy also avoids shared-FS open failures
    in docking worker processes.
    """
    src = Path(pdb_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f'Receptor PDB not found: {src}')
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    for line in src.read_text(encoding='utf-8', errors='replace').splitlines(keepends=True):
        record = line[0:6].strip().upper() if len(line) >= 4 else ''
        if record in {'MODEL', 'ENDMDL'}:
            continue
        kept.append(line)
    if not any(l.startswith(('ATOM', 'HETATM')) for l in kept):
        raise ValueError(f'No ATOM/HETATM records after normalizing {src}')
    out.write_text(''.join(kept), encoding='utf-8')
    return out


def prepare_apo_receptor_for_docking(receptor_pdb, ref_ligand, work_dir, apo_name='receptor_apo.pdb'):
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    apo_path = work / apo_name
    keys = ligand_residue_keys_from_mol(ref_ligand) if ref_ligand is not None else frozenset()
    if keys:
        strip_ligand_from_pdb(receptor_pdb, keys, apo_path)
    else:
        # Still stage a local copy so gnina never depends on the shared path.
        normalize_pdb_for_gnina(receptor_pdb, apo_path)
        return apo_path
    # Strip MODEL wrappers from the apo file in-place.
    normalize_pdb_for_gnina(apo_path, apo_path)
    return apo_path


def _het_residue_fragments(mol):
    groups = defaultdict(list)
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or not info.GetIsHeteroAtom():
            continue
        res = info.GetResidueName().strip()
        if res in PDB_SOLVENT_ION_RES:
            continue
        key = (info.GetChainId().strip(), info.GetResidueNumber(), res)
        groups[key].append(atom.GetIdx())

    frags = []
    for indices in groups.values():
        sub = _subset_mol_preserve_conformer(mol, indices)
        if sub is None:
            continue
        for frag in Chem.GetMolFrags(sub, asMols=True, sanitizeFrags=False):
            if _heavy_atom_count(frag) >= _MIN_LIGAND_HEAVY_ATOMS:
                frags.append(frag)
    return frags


def _select_fragments_for_smiles(frags, smiles):
    exact = [f for f in frags if mol_to_stereo_insensitive_key(f) == stereo_insensitive_smiles_key(smiles)]
    if exact:
        return exact
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        return []
    ref_heavy = ref.GetNumHeavyAtoms()
    ha_matches = [
        f for f in frags
        if abs(_heavy_atom_count(f) - ref_heavy) <= max(2, int(0.1 * ref_heavy))
    ]
    if len(ha_matches) == 1:
        return ha_matches
    if frags:
        return [max(frags, key=_heavy_atom_count)]
    return []


def extract_ligand_from_pdb(pdb_path, smiles=None):
    """Extract co-crystal ligand from a receptor PDB with 3D coordinates."""
    path = Path(pdb_path).expanduser()
    if not path.is_file():
        return None
    full = Chem.MolFromPDBFile(
        str(path),
        removeHs=False,
        sanitize=False,
        proximityBonding=True,
    )
    if full is None or full.GetNumConformers() == 0:
        return None

    frags = _het_residue_fragments(full)
    if not frags:
        return None

    if smiles:
        matches = _select_fragments_for_smiles(frags, smiles)
        if matches:
            ligand = max(matches, key=_heavy_atom_count)
            fixed = assign_bond_orders_from_smiles(ligand, smiles)
            return fixed if fixed is not None else ligand

    return max(frags, key=_heavy_atom_count)
