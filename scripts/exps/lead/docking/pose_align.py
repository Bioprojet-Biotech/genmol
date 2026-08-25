# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rigid alignment of SMILES conformers onto a reference co-crystal ligand."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS


def _kabsch_rigid_fit(p, q):
    if p.shape != q.shape or p.shape[0] < 3:
        raise ValueError('Kabsch needs matched (N,3) arrays with N>=3.')
    pc = p.mean(axis=0)
    qc = q.mean(axis=0)
    p0 = p - pc
    q0 = q - qc
    h = p0.T @ q0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = qc - r @ pc
    p_fit = (r @ p.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((p_fit - q) ** 2, axis=1))))
    return r, t, rms


def _apply_rigid_transform_to_mol(mol, rot, trans):
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        v = np.array([p.x, p.y, p.z], dtype=float)
        q = rot @ v + trans
        conf.SetAtomPosition(
            i,
            Chem.rdGeometry.Point3D(float(q[0]), float(q[1]), float(q[2])),
        )
    return out


def _find_heavy_atom_match_pairs(mol_query, mol_ref):
    ha_q = Chem.RemoveHs(Chem.Mol(mol_query))
    ha_r = Chem.RemoveHs(Chem.Mol(mol_ref))
    if ha_q.GetNumAtoms() < 3 or ha_r.GetNumAtoms() < 3:
        return []

    mcs = rdFMCS.FindMCS(
        [ha_q, ha_r],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        timeout=15,
    )
    if mcs.numAtoms >= 3:
        patt = Chem.MolFromSmarts(mcs.smartsString)
        if patt is not None:
            match_q = ha_q.GetSubstructMatch(patt)
            match_r = ha_r.GetSubstructMatch(patt)
            if len(match_q) >= 3 and len(match_q) == len(match_r):
                q_to_full = [a.GetIdx() for a in mol_query.GetAtoms() if a.GetAtomicNum() > 1]
                r_to_full = [a.GetIdx() for a in mol_ref.GetAtoms() if a.GetAtomicNum() > 1]
                return [(q_to_full[i], r_to_full[j]) for i, j in zip(match_q, match_r)]

    q_to_full = [a.GetIdx() for a in mol_query.GetAtoms() if a.GetAtomicNum() > 1]
    r_to_full = [a.GetIdx() for a in mol_ref.GetAtoms() if a.GetAtomicNum() > 1]
    if ha_q.GetNumAtoms() <= ha_r.GetNumAtoms():
        match = ha_r.GetSubstructMatch(ha_q)
        if len(match) >= 3:
            return [(q_to_full[i], r_to_full[j]) for i, j in enumerate(match)]
    else:
        match = ha_q.GetSubstructMatch(ha_r)
        if len(match) >= 3:
            return [(q_to_full[j], r_to_full[i]) for i, j in enumerate(match)]
    return []


def align_smiles_conformer_to_reference_ligand(smiles_mol, ref_ligand_mol):
    """Rigidly align smiles_mol onto ref_ligand_mol via MCS + Kabsch. Returns (mol, n_matched, rmsd)."""
    if smiles_mol is None or ref_ligand_mol is None:
        raise ValueError('Both molecules are required for alignment.')
    if smiles_mol.GetNumConformers() == 0 or ref_ligand_mol.GetNumConformers() == 0:
        raise ValueError('Both molecules need 3D coordinates.')

    pairs = _find_heavy_atom_match_pairs(smiles_mol, ref_ligand_mol)
    if len(pairs) < 3:
        raise ValueError(
            'Cannot align SMILES to co-crystal ligand: fewer than 3 heavy atoms share '
            'a common substructure.'
        )

    conf_q = smiles_mol.GetConformer()
    conf_r = ref_ligand_mol.GetConformer()
    p_pts = np.array(
        [[conf_q.GetAtomPosition(i).x, conf_q.GetAtomPosition(i).y, conf_q.GetAtomPosition(i).z]
         for i, _ in pairs],
        dtype=float,
    )
    q_pts = np.array(
        [[conf_r.GetAtomPosition(j).x, conf_r.GetAtomPosition(j).y, conf_r.GetAtomPosition(j).z]
         for _, j in pairs],
        dtype=float,
    )
    rot, trans, rms = _kabsch_rigid_fit(p_pts, q_pts)
    aligned = _apply_rigid_transform_to_mol(smiles_mol, rot, trans)
    return aligned, len(pairs), rms
