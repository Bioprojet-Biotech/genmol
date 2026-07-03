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

from time import time
import random
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem, QED, RDConfig
from scripts.exps.lead.docking.docking import DockingOracle
from scripts.exps.lead.docking.mol3d import load_mol_3d, smiles_from_mol
from genmol.sampler import Sampler
from genmol.utils.utils_chem import cut
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def uses_custom_docking(args):
    return args.dock_program != 'vina' or args.receptor_file is not None


def affinity_to_reward(affinities):
    return np.clip(-np.array(affinities), 0, None)


def load_fragments(path):
    """Load fragment SMILES from a text file (one per line)."""
    frags = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            frags.add(line)
    return frags


def validate_fragment_smiles(smiles, name):
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f'Invalid {name} SMILES: {smiles}')


class GenMolOpt():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.custom_docking = uses_custom_docking(args)

        script_dir = os.path.dirname(os.path.abspath(__file__))

        if self.custom_docking:
            self.predictor = DockingOracle(
                dock_program=self.args.dock_program,
                receptor_files=self.args.receptor_file,
                dock_binary=self.args.dock_binary,
                box_center=self.args.box_center,
                box_size=self.args.box_size,
                core_3d_file=self.args.core_3d,
                docking_ph=self.args.docking_ph,
                gnina_seed=self.args.seed,
                gnina_no_gpu=not self.args.gnina_gpu,
            )
            self.ds_column_names = (
                ['DS'] + [f'DS_{name}' for name in self.predictor.receptor_names[1:]]
                if len(self.predictor.receptor_names) > 1
                else ['DS']
            )
            self.start_smiles = self.args.start_smiles
            start_affinities = self.predictor.predict([self.start_smiles])
            self.start_affinity = start_affinities[self.predictor.primary_receptor_name][0]
            self.start_prop = float(affinity_to_reward([self.start_affinity])[0])
            self.start_ds_by_receptor = {
                name: float(affinity_to_reward(vals)[0])
                for name, vals in start_affinities.items()
            }
            print(f'Start prop: {self.start_prop}')
            print(f'Start affinity: {self.start_affinity}')
            run_label = (
                f'{self.predictor.receptor_names[0]}'
                f'_{self.args.dock_program}'
            )
        else:
            if self.args.oracle_name is None:
                self.args.oracle_name = 'parp1'
            df = pd.read_csv(os.path.join(script_dir, 'docking', 'actives.csv'))
            df = df[df['target'] == self.args.oracle_name]
            self.start_smiles = df['smiles'].iloc[self.args.start_mol_idx]
            self.start_prop = df['DS'].iloc[self.args.start_mol_idx]
            self.start_affinity = -self.start_prop
            self.predictor = DockingOracle(
                target=self.args.oracle_name,
                dock_program='vina',
            )
            self.ds_column_names = ['DS']
            run_label = f'{self.args.oracle_name}_id{self.args.start_mol_idx}'

        start_mol = Chem.MolFromSmiles(self.start_smiles)
        
        self.fpgen = AllChem.GetRDKitFPGenerator()
        self.start_fp = self.fpgen.GetFingerprint(start_mol)

        print(f'Start SMILES:\t{self.start_smiles}')
        print(f'Start DS:\t{self.start_prop}')

        self.init_frag = self.args.init_frag
        if self.init_frag:
            validate_fragment_smiles(self.init_frag, 'init_frag')

        self.population = self._build_initial_population()
        print(f'Initial population: {len(self.population)} frags')
        self.sampler = Sampler(self.args.model_path)

        self.fname = f'results/{run_label}_thr{self.args.sim_thr}_{self.args.seed}.csv'
        self.fname = os.path.join(ROOT_DIR, self.fname)
        base, ext = os.path.splitext(self.fname)
        self.pop_fname = f'{base}_population{ext}'
        print(f'\033[92m{self.fname}\033[0m')
        print(f'\033[92m{self.pop_fname}\033[0m')

        os.makedirs(os.path.dirname(self.fname), exist_ok=True)
        ds_header = ','.join(self.ds_column_names)
        with open(self.fname, 'wt') as f:
            f.write(f'SMILES,{ds_header},QED,SA,SIM,ref_SMILES\n')
            if self.custom_docking:
                start_ds = [
                    str(self.start_ds_by_receptor[name])
                    for name in self.predictor.receptor_names
                ]
            else:
                start_ds = [str(self.start_prop)]
            f.write(
                f'{self.start_smiles},{",".join(start_ds)},0,0,1,{self.start_smiles}\n'
            )
        with open(self.pop_fname, 'wt') as f:
            f.write('iteration,rank,DS,fragment\n')
        self.record_population(0)

    def _build_initial_population(self):
        frags = set()
        if self.args.fragments_file:
            frags |= load_fragments(self.args.fragments_file)
            print(f'Loaded {len(frags)} fragments from {self.args.fragments_file}')
        if self.args.fragments_file is None or self.args.self_fragment:
            start_frags = cut(self.start_smiles)
            print(f'Cut start_smiles into {len(start_frags)} fragments')
            frags |= start_frags
        if self.init_frag:
            frags.add(self.init_frag)
        if not frags:
            raise ValueError('Initial fragment population is empty')
        self.population = [(self.start_prop, frag) for frag in frags]
        self._dedupe_population()
        return self.population

    def _fragment_key(self, frag):
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return frag
        return Chem.MolToSmiles(mol)

    def _dedupe_population(self):
        best = {}
        for ds, frag in self.population:
            key = self._fragment_key(frag)
            if key not in best or ds > best[key][0]:
                best[key] = (ds, frag)
        self.population = sorted(best.values(), reverse=True)

    def reward_dock(self, smiles_list):
        affinities = self.predictor.predict(smiles_list)
        ds_by_receptor = {
            name: affinity_to_reward(vals)
            for name, vals in affinities.items()
        }
        primary = ds_by_receptor[self.predictor.primary_receptor_name]
        return primary, ds_by_receptor

    def reward_qed(self, mols):
        return [QED.qed(m) for m in mols]

    def reward_sa(self, mols):
        return [(10 - sascorer.calculateScore(m)) / 9 for m in mols]

    def reward_sim(self, mols):

        mol_fps = [self.fpgen.GetFingerprint(mol) for mol in mols]
        return DataStructs.BulkTanimotoSimilarity(self.start_fp, mol_fps)

    def reward(self, smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        rv, ds_by_receptor = self.reward_dock(smiles_list)
        rq = self.reward_qed(mols)
        rs = self.reward_sa(mols)
        rsim = self.reward_sim(mols)
        return rv, ds_by_receptor, rq, rs, rsim

    def attach(self, frag1, frag2):
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
        idx = np.random.randint(len(mols))
        return mols[idx][0]

    def attach_many(self, frags):
        mol = self.attach(frags[0], frags[1])
        for frag in frags[2:]:
            mol = self.attach(Chem.MolToSmiles(mol), frag)
        return mol

    def update_population(self, smiles_list, prop_list):
        rv_list, ds_by_receptor, rq_list, rs_list, rsim_list = prop_list
        for rv, rq, rs, rsim, smiles in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list):
            if rv > self.start_prop and rq >= 0.6 and rs >= 6/9 and rsim >= self.args.sim_thr:
                frags = {frag for frag in cut(smiles)}
                self.population.extend([(rv, frag) for frag in frags])
        self._dedupe_population()

    def generate(self):
        for _ in range(1000):
            pop_frags = [frag for prop, frag in self.population]
            frag1, frag2 = random.sample(pop_frags, 2)
            if self.init_frag:
                frags = [frag1, frag2, self.init_frag]
                random.shuffle(frags)
                mol = self.attach_many(frags)
            else:
                mol = self.attach(frag1, frag2)
            smiles = Chem.MolToSmiles(mol)
            if smiles is None: continue
            smiles = self.sampler.mask_modification(smiles, min_len=50, gamma=self.args.gamma)
            if smiles is not None:
                smiles = sorted(smiles.split('.'), key=len)[-1]     # get the largest
            return smiles

    def record_population(self, iteration):
        with open(self.pop_fname, 'a') as f:
            for rank, (ds, frag) in enumerate(self.population, start=1):
                f.write(f'{iteration},{rank},{ds},{frag}\n')

    def record(self, smiles_list, prop_list):
        rv_list, ds_by_receptor, rq_list, rs_list, rsim_list = prop_list
        with open(self.fname, 'a') as f:
            for i, smiles in enumerate(smiles_list):
                # if rsim_list[i] >= self.args.sim_thr:
                    ds_vals = [
                        str(ds_by_receptor[name][i])
                        for name in self.predictor.receptor_names
                    ]
                    f.write(
                        f'{smiles},{",".join(ds_vals)},{rq_list[i]},{rs_list[i]},'
                        f'{rsim_list[i]},{self.start_smiles}\n'
                    )

    def run(self):
        t_start = time()
        for i in range(self.args.num_iter):
            self.record_population(i + 1)
            smiles_list = [self.generate() for _ in range(self.args.num_gen)]
            prop_list = self.reward(smiles_list)
            self.update_population(smiles_list, prop_list)
            self.record(smiles_list, prop_list)
            print(f'[Iter {i+1:03d}] Top DS: {self.population[0][0]}')
        print(f'{time() - t_start:.2f} sec elapsed')


def validate_args(parser, args):
    custom = uses_custom_docking(args)
    if custom:
        missing = []
        if args.receptor_file is None:
            missing.append('--receptor_file')
        if args.box_center is None:
            missing.append('--box_center')
        if args.box_size is None:
            missing.append('--box_size')
        if args.start_smiles is None:
            missing.append('--start_smiles')
        if missing:
            parser.error(
                'Custom docking requires: ' + ', '.join(missing)
            )
        if args.oracle_name is not None:
            print('Note: --oracle_name is ignored when using custom docking.')
        if args.receptor_file:
            for path in args.receptor_file:
                if not os.path.exists(path):
                    parser.error(f'Receptor file not found: {path}')
    elif args.oracle_name is None:
        args.oracle_name = 'parp1'

    if args.fragments_file and not os.path.exists(args.fragments_file):
        parser.error(f'Fragments file not found: {args.fragments_file}')
    if args.core_3d:
        if not os.path.exists(args.core_3d):
            parser.error(f'Core 3D file not found: {args.core_3d}')
        try:
            core_mol = load_mol_3d(args.core_3d)
            if not args.init_frag:
                args.init_frag = smiles_from_mol(core_mol)
                print(f'Note: --init_frag derived from --core_3d: {args.init_frag}')
        except Exception as e:
            parser.error(f'Invalid --core_3d file: {e}')
    if args.init_frag:
        try:
            validate_fragment_smiles(args.init_frag, 'init_frag')
        except ValueError as e:
            parser.error(str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--oracle_name',
        type=str,
        default=None,
        choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2'],
        help='Preset benchmark target (bundled vina mode only)',
    )
    parser.add_argument(
        '-i', '--start_mol_idx',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Start molecule index in actives.csv (bundled vina mode only)',
    )
    parser.add_argument('-d', '--sim_thr',          type=float, default=0.4)
    parser.add_argument('-s', '--seed',             type=int,   default=0)
    parser.add_argument('-m', '--model_path',       type=str,   default='model.ckpt')
    parser.add_argument('--num_gen',                type=int,   default=100)
    parser.add_argument('--num_iter',               type=int,   default=10)
    parser.add_argument('--gamma',                  type=float, default=0)
    parser.add_argument(
        '--dock_program',
        type=str,
        default='vina',
        choices=['vina', 'gnina', 'unidock'],
        help='Docking backend used as the DS oracle (default: vina/qvina02)',
    )
    parser.add_argument(
        '--receptor_file',
        type=str,
        nargs='+',
        default=None,
        help='One or more aligned receptor PDB/PDBQT files; first is used for selection',
    )
    parser.add_argument(
        '--start_smiles',
        type=str,
        default=None,
        help='Starting molecule SMILES (required for custom docking; baseline DS is docked)',
    )
    parser.add_argument(
        '--dock_binary',
        type=str,
        default=None,
        help='Path to docking binary (default: bundled qvina02, or gnina/unidock on PATH)',
    )
    parser.add_argument(
        '--box_center',
        type=float,
        nargs=3,
        default=None,
        metavar=('X', 'Y', 'Z'),
        help='Docking box center in Angstrom (required for custom docking)',
    )
    parser.add_argument(
        '--box_size',
        type=float,
        nargs=3,
        default=None,
        metavar=('X', 'Y', 'Z'),
        help='Docking box size in Angstrom (required for custom docking)',
    )
    parser.add_argument(
        '--fragments_file',
        type=str,
        default=None,
        help='Text file with initial fragment SMILES, one per line',
    )
    parser.add_argument(
        '--self-fragment',
        action='store_true',
        help='Fragment start_smiles with cut() and add to the initial population',
    )
    parser.add_argument(
        '--init_frag',
        type=str,
        default=None,
        help='Mandatory fragment SMILES merged with two sampled fragments at each generation',
    )
    parser.add_argument(
        '--core_3d',
        type=str,
        default=None,
        help=(
            'SDF/MOL/PDB/PDBQT with a 3D substructure pose; '
            'ligands are placed via RDKit ConstrainedEmbed before docking. '
            'If --init_frag is omitted, SMILES are derived from this file'
        ),
    )
    parser.add_argument(
        '--docking_ph',
        type=float,
        default=7.4,
        help='Protonation pH for gnina ligand prep (default: 7.4)',
    )
    parser.add_argument(
        '--gnina_gpu',
        action='store_true',
        help='Enable GPU acceleration for gnina local minimize (default: CPU only)',
    )
    args = parser.parse_args()
    validate_args(parser, args)

    GenMolOpt(args).run()
