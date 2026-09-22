# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GenMol substructure inpainting — FLOWR-like local remask of a ligand region.

Masks SAFE fragments that overlap ``--substructure`` (heavy-atom indices or
SMILES/SMARTS of the region to *change*), then samples completions with GenMol.

CLI mirrors FLOWR.root local inpainting:
  https://github.com/jakiw/FLOWR / ~/repos/flowr_root README
  (--substructure_inpainting, --substructure, --filter_cond_substructure).

Atom indices are 0-based heavy-atom RDKit indices (hydrogens removed), matching
FLOWR ``scripts/draw_atom_numbers.py``.

Examples:
  python scripts/exps/inpaint/run.py \\
    --ref_ligand ligand.sdf \\
    --substructure_inpainting \\
    --substructure 10 11 12 13 \\
    --n_samples 50 \\
    --output results/inpaint.csv

  python scripts/exps/inpaint/run.py \\
    --ref_ligand ligand.sdf \\
    --substructure_inpainting \\
    --substructure c1ccccc1 \\
    --filter_cond_substructure \\
    --output results/inpaint.csv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.realpath('.'))
sys.path.insert(0, os.path.join(os.path.realpath('.'), 'src'))

import yaml
import pandas as pd
from rdkit import Chem, RDLogger

from genmol.sampler import Sampler

RDLogger.DisableLog('rdApp.*')

try:
    from tdc import Oracle
    oracle_qed = Oracle('qed')
    oracle_sa = Oracle('sa')
    _HAS_TDC = True
except Exception:
    _HAS_TDC = False


def parse_substructure(value: str):
    """Parse one --substructure token as int (atom index) or SMILES/SMARTS string."""
    try:
        return int(value)
    except ValueError:
        return value


def load_ref_mol(path: str, record: int = 0) -> Chem.Mol:
    """Load ligand as heavy-atom mol; preserve SDF heavy-atom order for indexing."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Ref ligand file not found: {path}')

    ext = os.path.splitext(path)[1].lower()

    if ext in ('.smi', '.txt', '.smiles'):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    mol = Chem.MolFromSmiles(line.split()[0])
                    if mol is None:
                        raise ValueError(f'Invalid SMILES in {path}')
                    return Chem.RemoveHs(mol)
        raise ValueError(f'No SMILES found in {path}')

    if ext == '.csv':
        df = pd.read_csv(path)
        col = 'smiles' if 'smiles' in df.columns else df.columns[0]
        mol = Chem.MolFromSmiles(str(df[col].iloc[0]).strip())
        if mol is None:
            raise ValueError(f'Invalid SMILES in {path}')
        return Chem.RemoveHs(mol)

    if ext in ('.mol', '.sdf', '.sd'):
        if ext == '.mol':
            mol = Chem.MolFromMolFile(path, removeHs=True)
        else:
            suppl = Chem.SDMolSupplier(path, removeHs=True)
            mol = None
            for idx, candidate in enumerate(suppl):
                if idx == record:
                    mol = candidate
                    break
        if mol is None:
            raise ValueError(f'Could not read molecule from {path} (record={record})')
        return Chem.RemoveHs(mol)

    # Fallback: first non-empty token as SMILES
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                mol = Chem.MolFromSmiles(line.split()[0])
                if mol is None:
                    raise ValueError(f'Invalid SMILES in {path}')
                return Chem.RemoveHs(mol)
    raise ValueError(f'No molecule found in {path}')


def resolve_mask_atom_indices(mol: Chem.Mol, substructure) -> list[int]:
    """Resolve --substructure to heavy-atom indices on *mol* (atoms to change)."""
    if not substructure:
        raise ValueError('--substructure is required with --substructure_inpainting')

    if all(isinstance(v, int) for v in substructure):
        n = mol.GetNumAtoms()
        indices = list(substructure)
        bad = [i for i in indices if i < 0 or i >= n]
        if bad:
            raise ValueError(
                f'Atom indices out of range for molecule with {n} heavy atoms: {bad}'
            )
        return indices

    if len(substructure) != 1 or not isinstance(substructure[0], str):
        raise ValueError(
            '--substructure must be either space-separated atom indices '
            "(e.g. '10 11 12') or a single SMILES/SMARTS string"
        )

    query_str = substructure[0]
    query = Chem.MolFromSmarts(query_str)
    if query is None:
        query = Chem.MolFromSmiles(query_str)
    if query is None or query.GetNumAtoms() == 0:
        raise ValueError(f'Could not parse --substructure as SMILES/SMARTS: {query_str!r}')

    matches = mol.GetSubstructMatches(query)
    if not matches:
        raise ValueError(f'--substructure pattern not found in reference ligand: {query_str!r}')
    if len(matches) > 1:
        print(
            f'Warning: {len(matches)} matches for --substructure; using the first.',
            file=sys.stderr,
        )
    return list(matches[0])


def remap_indices_to_smiles_mol(file_mol: Chem.Mol, smiles_mol: Chem.Mol, indices: list[int]) -> list[int]:
    """Remap heavy-atom indices from file-order mol onto MolFromSmiles(canonical) mol."""
    if file_mol.GetNumAtoms() != smiles_mol.GetNumAtoms():
        raise ValueError(
            'Atom-count mismatch between file mol and SMILES mol; cannot remap indices'
        )
    match = smiles_mol.GetSubstructMatch(file_mol)
    if not match or len(match) != file_mol.GetNumAtoms():
        # Fall back: assume same order (rare for non-canonical file SMILES)
        match = tuple(range(file_mol.GetNumAtoms()))
        if Chem.MolToSmiles(file_mol) != Chem.MolToSmiles(smiles_mol):
            raise ValueError(
                'Could not isomorphism-map file atom indices onto GenMol SMILES mol'
            )
    return [match[i] for i in indices]


def extract_keep_smiles(mol: Chem.Mol, mask_indices: list[int]) -> list[str]:
    """SMILES of connected components remaining after deleting mask atoms (keep region)."""
    mask = set(mask_indices)
    keep_idx = [i for i in range(mol.GetNumAtoms()) if i not in mask]
    if not keep_idx:
        return []
    rw = Chem.RWMol(Chem.Mol(mol))
    for idx in sorted(mask, reverse=True):
        rw.RemoveAtom(idx)
    keep = rw.GetMol()
    try:
        Chem.SanitizeMol(keep)
    except Exception:
        pass
    smi = Chem.MolToSmiles(keep)
    if not smi:
        return []
    parts = []
    for part in smi.split('.'):
        if part and Chem.MolFromSmiles(part) is not None:
            parts.append(part)
    return parts


def main():
    parser = argparse.ArgumentParser(
        description=(
            'GenMol local inpainting: remask SAFE fragments overlapping a '
            'substructure (FLOWR-compatible --substructure_inpainting CLI).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Atom indices are 0-based heavy-atom RDKit indices (Hs removed). '
            'Use FLOWR scripts/draw_atom_numbers.py to visualize numbering:\n'
            '  https://github.com/jakiw/FLOWR / ~/repos/flowr_root/scripts/draw_atom_numbers.py'
        ),
    )
    parser.add_argument(
        '--ref_ligand', type=str, required=True,
        help='Reference ligand (.sdf/.mol/.smi/.csv). Heavy-atom order defines --substructure indices.',
    )
    parser.add_argument(
        '--record', type=int, default=0,
        help='SDF record index when --ref_ligand has multiple molecules (default: 0).',
    )
    parser.add_argument(
        '--substructure_inpainting', action='store_true', required=True,
        help='Enable substructure inpainting (required; mirrors FLOWR).',
    )
    parser.add_argument(
        '--substructure', type=parse_substructure, nargs='+', required=True,
        help=(
            "Atoms to change: space-separated heavy-atom indices (e.g. '10 11 12') "
            'or one SMILES/SMARTS of the region to replace.'
        ),
    )
    parser.add_argument(
        '--filter_cond_substructure', action='store_true',
        help='Keep only samples that still contain the complementary (non-masked) region.',
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output CSV path.',
    )
    parser.add_argument(
        '--n_samples', type=int, default=None,
        help='Number of samples (overrides config num_samples).',
    )
    parser.add_argument(
        '-c', '--config', type=str, default='hparams.yaml',
        help='YAML config (default: hparams.yaml next to this script).',
    )
    parser.add_argument(
        '--mask_one', action='store_true',
        help='Remask a single random overlapping SAFE fragment instead of all of them.',
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    config = yaml.safe_load(open(config_path))
    hp = dict(config.get('inpainting') or {})
    mask_all = not args.mask_one and bool(hp.pop('mask_all', True))
    num_samples = args.n_samples if args.n_samples is not None else config['num_samples']

    file_mol = load_ref_mol(args.ref_ligand, record=args.record)
    mask_on_file = resolve_mask_atom_indices(file_mol, args.substructure)

    smiles = Chem.MolToSmiles(file_mol)
    smiles_mol = Chem.MolFromSmiles(smiles)
    if smiles_mol is None:
        raise ValueError(f'Could not re-parse reference SMILES: {smiles}')
    mask_indices = remap_indices_to_smiles_mol(file_mol, smiles_mol, mask_on_file)

    keep_smiles = None
    if args.filter_cond_substructure:
        keep_smiles = extract_keep_smiles(smiles_mol, mask_indices)
        if not keep_smiles:
            print(
                'Warning: empty keep region after masking; --filter_cond_substructure disabled.',
                file=sys.stderr,
            )
            keep_smiles = None

    print(f'Reference SMILES: {smiles}')
    print(f'Mask heavy-atom indices (GenMol mol): {mask_indices}')
    if keep_smiles:
        print(f'Keep filter SMILES: {keep_smiles}')

    sampler = Sampler(config['model_path'])
    samples = sampler.inpaint(
        smiles,
        mask_atom_indices=mask_indices,
        mol=smiles_mol,
        num_samples=num_samples,
        mask_all=mask_all,
        keep_smiles=keep_smiles,
        **hp,
    )
    # Unique while preserving order
    seen = set()
    unique = []
    for s in samples:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    samples = unique

    out_df = pd.DataFrame({
        'smiles': samples,
        'ref_smiles': [smiles] * len(samples),
        'mask_atoms': [','.join(str(i) for i in mask_indices)] * len(samples),
    })
    if _HAS_TDC and samples:
        out_df['qed'] = oracle_qed(samples)
        out_df['sa'] = oracle_sa(samples)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f'Wrote {len(samples)} samples to {args.output}')


if __name__ == '__main__':
    main()
