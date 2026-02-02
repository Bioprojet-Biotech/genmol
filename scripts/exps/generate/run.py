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


import os
import sys

sys.path.append(os.path.realpath('.'))

import argparse
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

TASKS = ('motif_extension', 'scaffold_decoration', 'superstructure_generation')


def load_ligand_smiles(path: str) -> str:
    """Load a single SMILES from a ligand file (.smi, .csv, .mol, .sdf)."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ref ligand file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in ('.smi', '.txt'):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return line.split()[0]
        raise ValueError(f"No SMILES found in {path}")

    if ext == '.csv':
        df = pd.read_csv(path)
        col = 'smiles' if 'smiles' in df.columns else df.columns[0]
        return str(df[col].iloc[0]).strip()

    if ext in ('.mol', '.sdf'):
        if ext == '.mol':
            mol = Chem.MolFromMolFile(path)
        else:
            suppl = Chem.SDMolSupplier(path)
            mol = suppl[0] if suppl else None
        if mol is None:
            raise ValueError(f"Could not read molecule from {path}")
        return Chem.MolToSmiles(mol)

    # Fallback: treat as text and take first line
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return line.split()[0]
    raise ValueError(f"No SMILES found in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GenMol generation (motif_extension, scaffold_decoration, superstructure_generation)."
    )
    parser.add_argument('--protein', type=str, default=None,
                        help="Target PDB file (for pipeline compatibility; not used by sampler).")
    parser.add_argument('--ref_ligand', type=str, required=True,
                        help="Target ligand file (SMILES or .smi/.csv/.mol/.sdf).")
    parser.add_argument('--output', type=str, required=True,
                        help="Output CSV path (e.g. ${DESTINATION_FOLDER}/output.csv).")
    parser.add_argument('--n_samples', type=int, default=None,
                        help="Number of samples to generate (overrides config num_samples).")
    parser.add_argument('--task', type=str, choices=TASKS, required=True,
                        help="Task: motif_extension, scaffold_decoration, or superstructure_generation.")
    parser.add_argument('-c', '--config', type=str, default='hparams.yaml',
                        help="YAML config file (default: hparams.yaml in script dir).")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, args.config)
    config = yaml.safe_load(open(config_path))

    num_samples = args.n_samples if args.n_samples is not None else config['num_samples']
    task = args.task

    if task not in config:
        raise ValueError(f"Config must contain section '{task}' (check {config_path}).")

    sampler = Sampler(config['model_path'])
    fragment = load_ligand_smiles(args.ref_ligand)

    samples = sampler.fragment_completion(
        fragment, num_samples=num_samples, **config[task]
    )
    samples = [s for s in samples if s]

    out_df = pd.DataFrame({
        'smiles': samples,
        'ref_smiles': [fragment] * len(samples),
    })
    if _HAS_TDC and samples:
        out_df['qed'] = oracle_qed(samples)
        out_df['sa'] = oracle_sa(samples)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == '__main__':
    main()
