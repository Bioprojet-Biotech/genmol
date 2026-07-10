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

"""Extract a SMILES substructure from an SDF, preserving 3D coordinates."""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from scripts.exps.lead.docking.mol3d import (
    extract_substructure_3d,
    load_mol_3d_record,
    write_mol_sdf,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Extract atoms matching a SMILES substructure from a 3D molecule file '
            'and write them to an output SDF at the original coordinates.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python scripts/exps/lead/extract_substructure_sdf.py \\\n'
            '    -i docked_ligand.sdf -s "c1ccccc1" -o core.sdf\n'
            '  python scripts/exps/lead/extract_substructure_sdf.py \\\n'
            '    -i poses.sdf -s "[*]c1ccccc1" -o fragment.sdf --record 2\n'
        ),
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input 3D structure file (SDF, MOL, PDB, or PDBQT).',
    )
    parser.add_argument(
        '-s', '--smiles',
        required=True,
        help='SMILES substructure to extract (heavy-atom match).',
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output SDF path for the extracted substructure.',
    )
    parser.add_argument(
        '--record',
        type=int,
        default=0,
        help='0-based SDF record index when the input has multiple molecules (default: 0).',
    )
    parser.add_argument(
        '--conf-id',
        type=int,
        default=0,
        help='Conformer id within the selected record (default: 0).',
    )
    parser.add_argument(
        '--match-index',
        type=int,
        default=0,
        help='Which substructure match to use when several are found (default: 0).',
    )
    parser.add_argument(
        '--no-chirality',
        action='store_true',
        help='Match substructure without requiring stereochemistry.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise SystemExit(f'Input file not found: {args.input}')

    mol = load_mol_3d_record(args.input, record=args.record, conf_id=args.conf_id)
    try:
        extracted = extract_substructure_3d(
            mol,
            args.smiles,
            match_index=args.match_index,
            conf_id=0,
            use_chirality=not args.no_chirality,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    write_mol_sdf(extracted, args.output)
    print(
        f'Wrote {extracted.GetNumAtoms()} atoms to {args.output}',
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
