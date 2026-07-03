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


# This file has been modified from MOOD.
#
# Source:
# https://github.com/SeulLee05/MOOD/blob/main/scorer/docking.py
#
# The license for the original version of this file can be
# found in LICENSE/3rd_party/LICENSE_MOOD.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import os
from shutil import rmtree
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import subprocess
from openbabel import pybel

from .mol3d import (
    build_initial_mol_for_gnina,
    constrained_embed_smiles,
    load_mol_3d,
    write_mol_file,
    write_mol_sdf,
)
from .ligand_prep import DEFAULT_DOCKING_PH


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

TARGET_BOX_PRESETS = {
    'fa7': {
        'box_center': (10.131, 41.879, 32.097),
        'box_size': (20.673, 20.198, 21.362),
    },
    'parp1': {
        'box_center': (26.413, 11.282, 27.238),
        'box_size': (18.521, 17.479, 19.995),
    },
    '5ht1b': {
        'box_center': (-26.602, 5.277, 17.898),
        'box_size': (22.5, 22.5, 22.5),
    },
    'jak2': {
        'box_center': (114.758, 65.496, 11.345),
        'box_size': (19.033, 17.929, 20.283),
    },
    'braf': {
        'box_center': (84.194, 6.949, -7.081),
        'box_size': (22.032, 19.211, 14.106),
    },
}

DEFAULT_DOCK_BINARIES = {
    'vina': os.path.join(ROOT_DIR, 'docking/qvina02'),
    'gnina': 'gnina',
    'unidock': 'unidock',
}

GNINA_SCORE_FIELDS = ('minimizedAffinity', 'CNNaffinity', 'affinity')


def parse_gnina_sdf_affinity(sdf_path):
    """Read gnina minimize score from the first valid pose in an output SDF."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        for field in GNINA_SCORE_FIELDS:
            if mol.HasProp(field):
                try:
                    return [float(mol.GetProp(field))]
                except ValueError:
                    continue
    return []


def parse_vina_style_affinity(result):
    """Parse affinity values from Vina-style docking output tables."""
    result_lines = result.split('\n')
    check_result = False
    affinity_list = []
    for result_line in result_lines:
        if result_line.startswith('-----+'):
            check_result = True
            continue
        if not check_result:
            continue
        if result_line.startswith('Writing output'):
            break
        if result_line.startswith('Refine time'):
            break
        parts = result_line.strip().split()
        if not parts or not parts[0].isdigit():
            break
        affinity_list.append(float(parts[1]))
    return affinity_list


class DockingOracle(object):
    def __init__(
        self,
        dock_program='vina',
        receptor_file=None,
        receptor_files=None,
        dock_binary=None,
        box_center=None,
        box_size=None,
        target=None,
        core_3d_file=None,
        docking_ph=DEFAULT_DOCKING_PH,
        gnina_seed=181129,
        gnina_no_gpu=True,
    ):
        super().__init__()
        self.target = target
        self.dock_program = dock_program

        if box_center is None or box_size is None:
            if target is None or target not in TARGET_BOX_PRESETS:
                raise ValueError(
                    'Provide --box_center and --box_size, or a preset --oracle_name target.'
                )
            preset = TARGET_BOX_PRESETS[target]
            box_center = box_center or preset['box_center']
            box_size = box_size or preset['box_size']
        self.box_center = tuple(box_center)
        self.box_size = tuple(box_size)

        if receptor_files is not None:
            files = list(receptor_files)
        elif receptor_file is not None:
            files = list(receptor_file) if isinstance(receptor_file, (list, tuple)) else [receptor_file]
        elif target is not None:
            files = [os.path.join(ROOT_DIR, f'docking/{target}.pdbqt')]
        else:
            raise ValueError('Provide --receptor_file or a preset --oracle_name target.')

        self.receptor_files = [os.path.abspath(f) for f in files]
        for path in self.receptor_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f'Receptor file not found: {path}')
        self.receptor_names = [
            os.path.splitext(os.path.basename(path))[0] for path in self.receptor_files
        ]
        self.primary_receptor_name = self.receptor_names[0]

        self.dock_binary = dock_binary or DEFAULT_DOCK_BINARIES[dock_program]
        self.exhaustiveness = 1
        self.num_sub_proc = 10
        self.num_cpu_dock = 5
        self.num_modes = 10
        self.timeout_gen3d = 30
        self.timeout_dock = 100
        self.core_3d_file = os.path.abspath(core_3d_file) if core_3d_file else None
        self._core_mol_cache = None
        self.docking_ph = docking_ph
        self.gnina_seed = gnina_seed
        self.gnina_no_gpu = gnina_no_gpu

        i = 0
        while True:
            tmp_dir = os.path.join(ROOT_DIR, f'docking/tmp/tmp{i}')
            if not os.path.exists(tmp_dir):
                print(f'Docking tmp dir: {tmp_dir}')
                os.makedirs(tmp_dir)
                self.temp_dir = tmp_dir
                break
            i += 1

        self._receptors_for_docking = [
            self._prepare_receptor(path, idx) for idx, path in enumerate(self.receptor_files)
        ]
        if self.core_3d_file:
            core = self._get_core_mol()
            print(
                f'Docking oracle: program={self.dock_program}, '
                f'binary={self.dock_binary}, receptors={self.receptor_names}, '
                f'core_3d={self.core_3d_file} ({core.GetNumAtoms()} atoms)'
            )
        elif self.dock_program == 'gnina':
            print(
                f'Docking oracle: program=gnina (local --minimize), '
                f'binary={self.dock_binary}, receptors={self.receptor_names}, '
                f'ph={self.docking_ph}'
            )
        else:
            print(
                f'Docking oracle: program={self.dock_program}, '
                f'binary={self.dock_binary}, receptors={self.receptor_names}'
            )

    def _prepare_receptor(self, receptor_file, receptor_idx=0):
        """Return a receptor path suitable for the selected docking backend."""
        ext = os.path.splitext(receptor_file)[1].lower()
        if self.dock_program in {'gnina', 'unidock'} or ext == '.pdbqt':
            return receptor_file

        receptor_pdbqt = os.path.join(self.temp_dir, f'receptor_{receptor_idx}.pdbqt')
        run_line = f'obabel {receptor_file} -O {receptor_pdbqt}'
        subprocess.check_output(
            run_line.split(),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        return receptor_pdbqt

    def _get_core_mol(self):
        if self._core_mol_cache is None:
            if not self.core_3d_file:
                raise ValueError('core_3d_file is not configured')
            self._core_mol_cache = load_mol_3d(self.core_3d_file)
        return self._core_mol_cache

    def prepare_gnina_ligand_sdf(self, smi, ligand_sdf_file, seed=0):
        """Build protonated SDF input for gnina local minimize."""
        core_mol = self._get_core_mol() if self.core_3d_file else None
        box_center = None if self.core_3d_file else self.box_center
        mol = build_initial_mol_for_gnina(
            smi,
            core_mol=core_mol,
            box_center=box_center,
            random_seed=seed,
            ph=self.docking_ph,
        )
        write_mol_sdf(mol, ligand_sdf_file)

    def _build_gnina_minimize_command(self, receptor_file, ligand_sdf_file, output_sdf, seed):
        cx, cy, cz = self.box_center
        sx, sy, sz = self.box_size
        cmd = [
            self.dock_binary,
            '-r', receptor_file,
            '-l', ligand_sdf_file,
            '-o', output_sdf,
            '--center_x', str(cx),
            '--center_y', str(cy),
            '--center_z', str(cz),
            '--size_x', str(sx),
            '--size_y', str(sy),
            '--size_z', str(sz),
            '--minimize',
            '--seed', str(seed),
            '--exhaustiveness', '1',
            '--num_modes', '1',
            '--cnn_scoring', 'none',
        ]
        if self.gnina_no_gpu:
            cmd.append('--no_gpu')
        else:
            cmd.extend(['--device', '0'])
        return cmd

    def gnina_local_minimize(self, receptor_file, ligand_sdf_file, output_sdf, seed=0):
        """Run gnina local minimize from a prepared input pose; return affinity list."""
        cmd = self._build_gnina_minimize_command(
            receptor_file,
            ligand_sdf_file,
            output_sdf,
            seed,
        )
        subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_dock,
            universal_newlines=True,
        )
        affinities = parse_gnina_sdf_affinity(output_sdf)
        if affinities:
            return affinities
        return parse_vina_style_affinity('')

    def gen_3d(self, smi, ligand_mol_file, seed=0):
        """Generate initial 3D conformation from SMILES."""
        if self.core_3d_file:
            mol = constrained_embed_smiles(
                smi,
                self._get_core_mol(),
                random_seed=seed,
            )
            write_mol_file(mol, ligand_mol_file)
            return

        run_line = 'obabel -:%s --gen3D -O %s' % (smi, ligand_mol_file)
        subprocess.check_output(
            run_line.split(),
            stderr=subprocess.STDOUT,
            timeout=self.timeout_gen3d,
            universal_newlines=True,
        )

    def _build_dock_command(
        self,
        receptor_file,
        ligand_pdbqt_file,
        docking_output,
        output_dir=None,
    ):
        cx, cy, cz = self.box_center
        sx, sy, sz = self.box_size

        if self.dock_program == 'vina':
            cmd = [
                self.dock_binary,
                '--receptor', receptor_file,
                '--ligand', ligand_pdbqt_file,
                '--out', docking_output,
                '--center_x', str(cx),
                '--center_y', str(cy),
                '--center_z', str(cz),
                '--size_x', str(sx),
                '--size_y', str(sy),
                '--size_z', str(sz),
                '--cpu', str(self.num_cpu_dock),
                '--num_modes', str(self.num_modes),
                '--exhaustiveness', str(self.exhaustiveness),
            ]
        elif self.dock_program == 'gnina':
            raise ValueError('Use prepare_gnina_ligand_sdf + gnina_local_minimize for gnina')
        elif self.dock_program == 'unidock':
            cmd = [
                self.dock_binary,
                '--receptor', receptor_file,
                '--ligand', ligand_pdbqt_file,
                '--center_x', str(cx),
                '--center_y', str(cy),
                '--center_z', str(cz),
                '--size_x', str(sx),
                '--size_y', str(sy),
                '--size_z', str(sz),
                '--num_modes', str(self.num_modes),
                '--search_mode', 'balance',
                '--scoring', 'vina',
                '--dir', output_dir,
            ]
        else:
            raise ValueError(f'Unsupported dock_program: {self.dock_program}')
        return cmd

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_output, output_dir=None):
        """Run docking and return affinity values for the input molecule."""
        ms = list(pybel.readfile('mol', ligand_mol_file))
        m = ms[0]
        m.write('pdbqt', ligand_pdbqt_file, overwrite=True)

        cmd = self._build_dock_command(
            receptor_file,
            ligand_pdbqt_file,
            docking_output,
            output_dir=output_dir,
        )
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        result = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_dock,
            universal_newlines=True,
        )
        return parse_vina_style_affinity(result)

    def creator(self, q, data, num_sub_proc):
        """Put data to queue."""
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for _ in range(num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        """Generate subprocess for docking."""
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            dock_seed = self.gnina_seed + idx + sub_id if self.dock_program == 'gnina' else idx + sub_id
            try:
                if self.dock_program == 'gnina':
                    ligand_sdf_file = '%s/ligand_%s.sdf' % (self.temp_dir, sub_id)
                    self.prepare_gnina_ligand_sdf(smi, ligand_sdf_file, seed=dock_seed)
                else:
                    ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
                    self.gen_3d(smi, ligand_mol_file, seed=dock_seed)
            except Exception as exc:
                print(f'ligand prep unexpected error: {smi} ({exc})')
                return_dict[idx] = {name: 99.9 for name in self.receptor_names}
                continue

            affinities = {}
            for rec_idx, (receptor_name, receptor_file) in enumerate(
                zip(self.receptor_names, self._receptors_for_docking)
            ):
                if self.dock_program == 'gnina':
                    docking_output = '%s/dock_%s_%s.sdf' % (self.temp_dir, sub_id, rec_idx)
                    try:
                        affinity_list = self.gnina_local_minimize(
                            receptor_file,
                            ligand_sdf_file,
                            docking_output,
                            seed=dock_seed,
                        )
                    except Exception:
                        print(f'gnina minimize unexpected error ({receptor_name}): {smi}')
                        affinities[receptor_name] = 99.9
                        continue
                else:
                    ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
                    ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
                    docking_output = '%s/dock_%s_%s.pdbqt' % (self.temp_dir, sub_id, rec_idx)
                    output_dir = '%s/dock_out_%s_%s' % (self.temp_dir, sub_id, rec_idx)
                    try:
                        affinity_list = self.docking(
                            receptor_file,
                            ligand_mol_file,
                            ligand_pdbqt_file,
                            docking_output,
                            output_dir=output_dir,
                        )
                    except Exception:
                        print(f'docking unexpected error ({receptor_name}): {smi}')
                        affinities[receptor_name] = 99.9
                        continue
                if len(affinity_list) == 0:
                    affinity_list.append(99.9)
                affinities[receptor_name] = affinity_list[0]

            return_dict[idx] = affinities

    def predict(self, smiles_list):
        """
        Input SMILES list.
        Output per-molecule affinities keyed by receptor name.
        If docking fails for a receptor, affinity is 99.9.
        """
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(
            target=self.creator,
            args=(q1, data, self.num_sub_proc),
        )
        proc_master.start()

        procs = []
        for sub_id in range(self.num_sub_proc):
            proc = Process(
                target=self.docking_subprocess,
                args=(q1, return_dict, sub_id),
            )
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()
        for proc in procs:
            proc.join()

        keys = sorted(return_dict.keys())
        per_receptor = {name: [] for name in self.receptor_names}
        for key in keys:
            affinities = return_dict[key]
            for name in self.receptor_names:
                per_receptor[name].append(affinities.get(name, 99.9))
        return per_receptor

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            rmtree(self.temp_dir)
            print(f'{self.temp_dir} removed')


# Backward-compatible alias
DockingVina = DockingOracle
