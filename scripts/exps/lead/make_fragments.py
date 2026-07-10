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

"""Generate fragment SMILES lists for lead optimization (--fragments_file input)."""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from rdkit import Chem

from genmol.utils.utils_chem import cut


def load_smiles_from_file(path):
    """Load SMILES from a text file (one per line; # comments and blank lines skipped)."""
    smiles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            smiles.append(line)
    return smiles


def canonical_fragment(frag):
    mol = Chem.MolFromSmiles(frag)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def collect_fragments(smiles_list, init_frag=None):
    """Cut each input SMILES into fragments and return a sorted, deduplicated list."""
    frags = set()
    for smi in smiles_list:
        if Chem.MolFromSmiles(smi) is None:
            print(f'warning: invalid SMILES skipped: {smi}', file=sys.stderr)
            continue
        frags |= cut(smi)

    if init_frag:
        key = canonical_fragment(init_frag)
        if key is None:
            raise ValueError(f'Invalid init_frag SMILES: {init_frag}')
        frags.add(key)

    canonical = set()
    for frag in frags:
        key = canonical_fragment(frag)
        if key is not None:
            canonical.add(key)
    return sorted(canonical)


def write_fragments(fragments, output_path):
    if output_path == '-':
        for frag in fragments:
            print(frag)
        return
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        for frag in fragments:
            f.write(f'{frag}\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate fragment SMILES from input molecules for use with '
            'scripts/exps/lead/run.py --fragments_file.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python scripts/exps/lead/make_fragments.py -i molecules.txt -o frags.txt\n'
            '  python scripts/exps/lead/make_fragments.py "CCO" "c1ccccc1" -o frags.txt\n'
            '  python scripts/exps/lead/make_fragments.py -i molecules.txt --init_frag "[*]c1ccccc1"\n'
        ),
    )
    parser.add_argument(
        'smiles',
        nargs='*',
        help='Input SMILES (one or more).',
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Text file with input SMILES, one per line (# comments allowed).',
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='-',
        help='Output fragment list path (default: stdout).',
    )
    parser.add_argument(
        '--init_frag',
        type=str,
        default=None,
        help='Optional fragment SMILES to include in the output list.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    smiles_list = list(args.smiles)
    if args.input:
        if not os.path.exists(args.input):
            raise SystemExit(f'Input file not found: {args.input}')
        smiles_list.extend(load_smiles_from_file(args.input))

    if not smiles_list:
        raise SystemExit('Provide SMILES on the command line and/or via --input.')

    try:
        fragments = collect_fragments(smiles_list, init_frag=args.init_frag)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if not fragments:
        raise SystemExit('No fragments generated from the provided SMILES.')

    write_fragments(fragments, args.output)
    if args.output != '-':
        print(
            f'Wrote {len(fragments)} fragments from {len(smiles_list)} molecule(s) to {args.output}',
            file=sys.stderr,
        )


if __name__ == '__main__':
    main()
