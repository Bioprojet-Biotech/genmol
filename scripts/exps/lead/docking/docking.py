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
import shutil
from shutil import rmtree
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import subprocess
from openbabel import pybel

from .mol3d import (
    build_initial_mol_for_gnina,
    constrained_embed_smiles,
    load_mol_3d,
    load_first_pose_from_sdf,
    write_mol_file,
    write_mol_sdf,
)
from .ligand_prep import DEFAULT_DOCKING_PH


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _subprocess_env_for_docking(hide_gpu=False):
    """Env for external docking binaries.

    Importing the Python OpenBabel bindings sets BABEL_LIBDIR/BABEL_DATADIR to the
    venv copy. Gnina ships its own OpenBabel; inheriting the venv paths makes it
    fail to open receptor PDBs with a misleading 'could not open ... for reading'.

    When *hide_gpu* is True, clear CUDA_VISIBLE_DEVICES so gnina cannot touch the
    GPU (even ``--no_gpu`` still probes CUDA on startup and can destabilize the
    driver when GenMol already holds a CUDA context).
    """
    env = os.environ.copy()
    env.pop('BABEL_LIBDIR', None)
    env.pop('BABEL_DATADIR', None)
    if hide_gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


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
        reference_smiles=None,
        num_sub_proc=None,
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
        # Gnina on GPU must stay single-process (GenMol + parallel gnina wedges the driver).
        # CPU gnina defaults to 1 worker; vina/unidock keep the historical 10.
        self.gnina_no_gpu = gnina_no_gpu
        if dock_program == 'gnina' and not gnina_no_gpu:
            self.num_sub_proc = 1
        elif num_sub_proc is not None:
            self.num_sub_proc = max(1, int(num_sub_proc))
        elif dock_program == 'gnina':
            self.num_sub_proc = 1
        else:
            self.num_sub_proc = 10
        self.num_cpu_dock = 5
        self.num_modes = 10
        self.timeout_gen3d = 30
        self.timeout_dock = 100
        self.core_3d_file = os.path.abspath(core_3d_file) if core_3d_file else None
        self._core_mol_cache = None
        self.docking_ph = docking_ph
        self.gnina_seed = gnina_seed
        self.reference_smiles = reference_smiles
        self._ref_ligand_mol = None

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
        if self.dock_program == 'gnina':
            self._prepare_gnina_receptors()
        if self.core_3d_file:
            core = self._get_core_mol()
            print(
                f'Docking oracle: program={self.dock_program}, '
                f'binary={self.dock_binary}, receptors={self.receptor_names}, '
                f'core_3d={self.core_3d_file} ({core.GetNumAtoms()} atoms)'
            )
        elif self.dock_program == 'gnina':
            ref_note = ''
            if self._ref_ligand_mol is not None:
                ref_note = ', co-crystal ligand from receptor PDB'
            addon_note = ''
            if len(self.receptor_names) > 1:
                addon_note = (
                    f', reference pose on {self.receptor_names[0]} -> pose_1, '
                    f'{len(self.receptor_names) - 1} addon receptor(s) reuse pose_1'
                )
            print(
                f'Docking oracle: program=gnina (local --minimize), '
                f'binary={self.dock_binary}, receptors={self.receptor_names}, '
                f'ph={self.docking_ph}, workers={self.num_sub_proc}, '
                f'gpu={"on" if not self.gnina_no_gpu else "off"}'
                f'{ref_note}{addon_note}'
            )
        else:
            print(
                f'Docking oracle: program={self.dock_program}, '
                f'binary={self.dock_binary}, receptors={self.receptor_names}'
            )

    def _prepare_receptor(self, receptor_file, receptor_idx=0):
        """Return a receptor path suitable for the selected docking backend."""
        ext = os.path.splitext(receptor_file)[1].lower()
        if self.dock_program == 'gnina' and ext in {'.pdb', '.ent'}:
            return receptor_file
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

    def _prepare_gnina_receptors(self):
        """Stage local gnina-ready apo PDBs (never pass shared paths to workers)."""
        from .receptor_prep import (
            extract_ligand_from_pdb,
            normalize_pdb_for_gnina,
            prepare_apo_receptor_for_docking,
        )

        prepared = []
        for idx, receptor_file in enumerate(self._receptors_for_docking):
            ext = os.path.splitext(receptor_file)[1].lower()
            if ext not in {'.pdb', '.ent'}:
                # Still copy non-PDB receptors next to the job to avoid shared-FS issues.
                local = os.path.join(self.temp_dir, f'receptor_{idx}{ext or ".pdbqt"}')
                if os.path.abspath(receptor_file) != os.path.abspath(local):
                    shutil.copy2(receptor_file, local)
                prepared.append(local)
                continue

            ref_ligand = extract_ligand_from_pdb(receptor_file, smiles=self.reference_smiles)
            if ref_ligand is not None and self._ref_ligand_mol is None:
                self._ref_ligand_mol = ref_ligand

            apo_name = f'receptor_{idx}_apo.pdb'
            if ref_ligand is not None:
                apo_path = prepare_apo_receptor_for_docking(
                    receptor_file,
                    ref_ligand,
                    self.temp_dir,
                    apo_name=apo_name,
                )
                print(
                    f'Gnina receptor: local apo ready '
                    f'({os.path.basename(receptor_file)} -> {os.path.basename(str(apo_path))})'
                )
            else:
                apo_path = normalize_pdb_for_gnina(
                    receptor_file,
                    os.path.join(self.temp_dir, apo_name),
                )
                print(
                    f'Gnina receptor: no co-crystal ligand extracted; '
                    f'staged normalized local copy -> {os.path.basename(str(apo_path))}'
                )
            prepared.append(str(apo_path))
        self._receptors_for_docking = prepared

    def _get_core_mol(self):
        if self._core_mol_cache is None:
            if not self.core_3d_file:
                raise ValueError('core_3d_file is not configured')
            self._core_mol_cache = load_mol_3d(self.core_3d_file)
        return self._core_mol_cache

    def prepare_gnina_ligand_sdf(self, smi, ligand_sdf_file, seed=0):
        """Build protonated SDF input for gnina local minimize."""
        core_mol = self._get_core_mol() if self.core_3d_file else None
        ref_ligand = None if self.core_3d_file else self._ref_ligand_mol
        box_center = None
        if core_mol is None and ref_ligand is None:
            box_center = self.box_center
        mol = build_initial_mol_for_gnina(
            smi,
            core_mol=core_mol,
            ref_ligand_mol=ref_ligand,
            box_center=box_center,
            random_seed=seed,
            ph=self.docking_ph,
        )
        write_mol_sdf(mol, ligand_sdf_file)

    def _use_pose_as_gnina_input(self, pose_sdf, ligand_sdf_file):
        """Copy a reference pose SDF to serve as gnina ligand input."""
        load_first_pose_from_sdf(pose_sdf)
        shutil.copy2(pose_sdf, ligand_sdf_file)

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
            env=_subprocess_env_for_docking(hide_gpu=self.gnina_no_gpu),
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
            ligand_sdf_file = '%s/ligand_%s.sdf' % (self.temp_dir, sub_id)
            try:
                if self.dock_program == 'gnina':
                    self.prepare_gnina_ligand_sdf(smi, ligand_sdf_file, seed=dock_seed)
                else:
                    ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
                    self.gen_3d(smi, ligand_mol_file, seed=dock_seed)
            except Exception as exc:
                print(f'ligand prep unexpected error: {smi} ({exc})')
                return_dict[idx] = {name: 99.9 for name in self.receptor_names}
                continue

            affinities = {}
            reference_pose_sdf = None
            for rec_idx, (receptor_name, receptor_file) in enumerate(
                zip(self.receptor_names, self._receptors_for_docking)
            ):
                if self.dock_program == 'gnina':
                    if rec_idx == 0:
                        # Reference protein: build initial placement, minimize -> pose_1
                        docking_output = '%s/pose_1_%s.sdf' % (self.temp_dir, sub_id)
                        current_ligand_input = ligand_sdf_file
                    else:
                        # Addon receptors: reuse pose_1 from reference protein
                        docking_output = '%s/dock_%s_%s.sdf' % (
                            self.temp_dir, sub_id, rec_idx,
                        )
                        if reference_pose_sdf is None:
                            print(
                                f'addon skip ({receptor_name}): {smi} '
                                f'(reference pose_1 unavailable)'
                            )
                            affinities[receptor_name] = 99.9
                            continue
                        addon_input = '%s/ligand_%s_rec%s.sdf' % (
                            self.temp_dir, sub_id, rec_idx,
                        )
                        try:
                            self._use_pose_as_gnina_input(
                                reference_pose_sdf, addon_input,
                            )
                            current_ligand_input = addon_input
                        except Exception as exc:
                            print(
                                f'pose_1 reuse failed ({receptor_name}): '
                                f'{smi} ({exc})'
                            )
                            affinities[receptor_name] = 99.9
                            continue
                    try:
                        affinity_list = self.gnina_local_minimize(
                            receptor_file,
                            current_ligand_input,
                            docking_output,
                            seed=dock_seed,
                        )
                    except Exception as exc:
                        detail = str(exc).strip() or type(exc).__name__
                        if isinstance(exc, subprocess.CalledProcessError) and exc.output:
                            # Keep the last non-empty gnina lines (often the real reason).
                            tail = [
                                ln for ln in str(exc.output).splitlines() if ln.strip()
                            ][-5:]
                            detail = ' | '.join(tail) if tail else detail
                        print(
                            f'gnina minimize unexpected error ({receptor_name}): '
                            f'{smi} ({detail})'
                        )
                        affinities[receptor_name] = 99.9
                        continue
                    if len(affinity_list) == 0:
                        affinity_list.append(99.9)
                    affinities[receptor_name] = affinity_list[0]
                    if rec_idx == 0 and affinities[receptor_name] < 99.9:
                        reference_pose_sdf = docking_output
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
        if self.num_sub_proc <= 1:
            return self._predict_serial(smiles_list)
        return self._predict_parallel(smiles_list)

    def _predict_serial(self, smiles_list):
        """Dock in-process (no fork). Safe after GenMol has initialized CUDA."""
        return_dict = {}
        q = _SerialQueue(list(enumerate(smiles_list)))
        self.docking_subprocess(q, return_dict, sub_id=0)
        keys = sorted(return_dict.keys())
        per_receptor = {name: [] for name in self.receptor_names}
        for key in keys:
            affinities = return_dict[key]
            for name in self.receptor_names:
                per_receptor[name].append(affinities.get(name, 99.9))
        return per_receptor

    def _predict_parallel(self, smiles_list):
        """Dock with worker processes.

        Uses the ``spawn`` start method so workers do not inherit a live CUDA
        context from the GenMol parent (fork-after-CUDA can kill the driver).
        """
        data = list(enumerate(smiles_list))
        ctx = multiprocessing.get_context('spawn')
        q1 = ctx.Queue()
        manager = ctx.Manager()
        return_dict = manager.dict()
        proc_master = ctx.Process(
            target=_docking_creator,
            args=(q1, data, self.num_sub_proc),
        )
        proc_master.start()

        procs = []
        worker_state = self._worker_state()
        for sub_id in range(self.num_sub_proc):
            proc = ctx.Process(
                target=_docking_worker,
                args=(worker_state, q1, return_dict, sub_id),
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

    def _worker_state(self):
        """Picklable snapshot for spawn workers (avoid sending live CUDA objects)."""
        return {
            'dock_program': self.dock_program,
            'dock_binary': self.dock_binary,
            'box_center': self.box_center,
            'box_size': self.box_size,
            'temp_dir': self.temp_dir,
            'timeout_dock': self.timeout_dock,
            'timeout_gen3d': self.timeout_gen3d,
            'num_cpu_dock': self.num_cpu_dock,
            'num_modes': self.num_modes,
            'exhaustiveness': self.exhaustiveness,
            'core_3d_file': self.core_3d_file,
            'docking_ph': self.docking_ph,
            'gnina_seed': self.gnina_seed,
            'gnina_no_gpu': self.gnina_no_gpu,
            'reference_smiles': self.reference_smiles,
            'receptor_names': list(self.receptor_names),
            'receptors_for_docking': list(self._receptors_for_docking),
        }

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            rmtree(self.temp_dir)
            print(f'{self.temp_dir} removed')


class _SerialQueue:
    """Minimal queue stand-in for serial docking (creator + one worker)."""

    def __init__(self, items):
        self._items = list(items) + ['DONE']

    def get(self):
        return self._items.pop(0)


def _docking_creator(q, data, num_sub_proc):
    for d in data:
        q.put((d[0], d[1]))
    for _ in range(num_sub_proc):
        q.put('DONE')


def _docking_worker(state, q, return_dict, sub_id):
    """Spawn-safe worker: rebuild a lightweight oracle from pickled state."""
    oracle = DockingOracle.__new__(DockingOracle)
    oracle.dock_program = state['dock_program']
    oracle.dock_binary = state['dock_binary']
    oracle.box_center = state['box_center']
    oracle.box_size = state['box_size']
    oracle.temp_dir = state['temp_dir']
    oracle.timeout_dock = state['timeout_dock']
    oracle.timeout_gen3d = state['timeout_gen3d']
    oracle.num_cpu_dock = state['num_cpu_dock']
    oracle.num_modes = state['num_modes']
    oracle.exhaustiveness = state['exhaustiveness']
    oracle.core_3d_file = state['core_3d_file']
    oracle._core_mol_cache = None
    oracle.docking_ph = state['docking_ph']
    oracle.gnina_seed = state['gnina_seed']
    oracle.gnina_no_gpu = state['gnina_no_gpu']
    oracle.reference_smiles = state['reference_smiles']
    oracle.receptor_names = state['receptor_names']
    oracle._receptors_for_docking = state['receptors_for_docking']
    oracle._ref_ligand_mol = None
    oracle.docking_subprocess(q, return_dict, sub_id=sub_id)


# Backward-compatible alias
DockingVina = DockingOracle
