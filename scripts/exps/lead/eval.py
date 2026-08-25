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


import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-d', '--sim_thr', type=float, default=0.4)
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    # Support both legacy (no iteration) and current headers.
    colmap = {c.lower(): c for c in df.columns}
    smiles_col = colmap.get('smiles')
    ds_col = colmap.get('ds') or colmap.get('dockingscore')
    qed_col = colmap.get('qed')
    sa_col = colmap.get('sa')
    sim_col = colmap.get('sim') or colmap.get('tanimoto_from_ref')
    if not all([smiles_col, ds_col, qed_col, sa_col, sim_col]):
        # Legacy files without a header row
        df = pd.read_csv(
            args.file,
            names=['smiles', 'DS', 'QED', 'SA', 'SIM', 'ref'],
            header=None,
        )
        smiles_col, ds_col, qed_col, sa_col, sim_col = 'smiles', 'DS', 'QED', 'SA', 'SIM'

    num_gen = 1000  # len(df)
    df = df.drop_duplicates(subset=[smiles_col])
    print(f'Uniqueness:\t{len(df) / num_gen}')

    df = df[df[sim_col] >= args.sim_thr]
    df = df[df[qed_col] >= 0.6]
    df = df[df[sa_col] >= 6 / 9]
    if not len(df):
        print('Lead optimization failed')
    else:
        df = df.sort_values(by=ds_col, ascending=False)
        print(f'Top DS:\t\t{(df[ds_col].iloc[0])}')
        print(f'Top mol:\t{(df[smiles_col].iloc[0])}')
