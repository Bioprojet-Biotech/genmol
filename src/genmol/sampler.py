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
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import itertools
import pickle
import torch
import random
import safe as sf
from rdkit import Chem
from genmol.utils.utils_chem import safe_to_smiles, filter_by_substructure, mix_sequences, Slicer
from genmol.utils.bracket_safe_converter import BracketSAFEConverter, bracketsafe2safe
from genmol.model import GenMol


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def load_model_from_path(path):
    model = GenMol.load_from_checkpoint(path)
    model.backbone.eval()
    if model.ema:
        model.ema.store(itertools.chain(model.backbone.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters()))
    return model


class Sampler:
    def __init__(self, path):
        self.model = load_model_from_path(path)
        self.slicer = Slicer()
        self.dot_index = self.model.tokenizer('.')['input_ids'][1]
        self.pad_index = self.model.tokenizer.pad_token_id
        self.mdlm = self.model.mdlm
        self.mdlm.to_device(self.model.device)
        
    @torch.no_grad()
    def generate(self, x, softmax_temp=1.2, randomness=2, fix=True, gamma=0, w=2, **kwargs):
        x = x.to(self.model.device)
        num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
        attention_mask = x != self.pad_index
        
        for i in range(num_steps):
            logits = self.model(x, attention_mask)

            if gamma and w:
                x_poor = x.clone()
                context_tokens = (x_poor[0] != self.model.bos_index).to(int) * \
                    (x_poor[0] != self.model.eos_index).to(int) * \
                    (x_poor[0] != self.model.mask_index).to(int) * \
                    (x_poor[0] != self.pad_index).to(int)
                context_token_ids = context_tokens.nonzero(as_tuple=True)[0].tolist()
                # mask 100 * gamma % of the context (given fragments) tokens
                num_mask_poor = int(context_tokens.sum() * gamma)
                mask_idx_poor = random.sample(context_token_ids, num_mask_poor)
                x_poor[:, mask_idx_poor] = self.model.mask_index
                logits_poor = self.model(x_poor, attention_mask=attention_mask)
                logits = w * logits + (1 - w) * logits_poor

            x = self.mdlm.step_confidence(logits, x, i, num_steps, softmax_temp, randomness)
            
        # decode to SAFE strings
        samples = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
        # convert to SMILES strings
        if self.model.config.training.get('use_bracket_safe'):
            samples = [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in samples]
        else:
            samples = [safe_to_smiles(s, fix=fix) for s in samples]
        # remove None and take the largest
        samples = [sorted(s.split('.'), key=len)[-1] for s in samples if s]
        return samples

    def _insert_mask(self, x, num_samples, min_add_len=18, **kwargs):
        with open(os.path.join(ROOT_DIR, 'data/len.pk'), 'rb') as f:
            seq_len_list = pickle.load(f)
        
        x = x[0]
        x_new = []
        for _ in range(num_samples):
            add_seq_len = max(random.choice(seq_len_list) - len(x), min_add_len)
            x_new.append(torch.hstack([x[:-1],
                                      torch.full((add_seq_len,), self.model.mask_index),
                                      x[-1:]]))
        pad_len = max([len(xx) for xx in x_new])
        x_new = [torch.hstack([xx,torch.full((pad_len - len(xx),), self.pad_index)]) for xx in x_new]
        return torch.stack(x_new)
    
    @torch.no_grad()
    def de_novo_generation(self, num_samples=1, softmax_temp=0.8, randomness=0.5, min_add_len=40, **kwargs):
        # Prepare Fully Masked Inputs
        x = torch.hstack([torch.full((1, 1), self.model.bos_index),
                          torch.full((1, 1), self.model.eos_index)])
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        x = x.to(self.model.device)
        return self.generate(x, softmax_temp, randomness)
    
    def fragment_linking_onestep(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0, min_add_len=30, **kwargs):
        if self.model.config.training.get('use_bracket_safe'):
            encoded_fragment = BracketSAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        else:
            encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        
        x = self.model.tokenizer([encoded_fragment + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        samples = self.generate(x, softmax_temp, randomness, gamma=gamma)
        samples = filter_by_substructure(samples, fragment)
        return samples
    
    def fragment_linking(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0, min_add_len=30, **kwargs):
        encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        prefix, suffix = encoded_fragment.split('.')

        x = self.model.tokenizer([prefix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        prefix_samples = self.generate(x, softmax_temp, randomness, gamma=gamma)

        x = self.model.tokenizer([suffix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        suffix_samples = self.generate(x, softmax_temp, randomness, gamma=gamma)
        
        samples = filter_by_substructure(mix_sequences(prefix_samples, suffix_samples,
                                                      *fragment.split('.'), num_samples), fragment)
        return samples
        
    def fragment_completion(self, fragment, num_samples=1, apply_filter=True, softmax_temp=1.2, randomness=2, gamma=0, **kwargs):
        if '*' not in fragment:     # superstructure generation
            cores = sf.utils.list_individual_attach_points(Chem.MolFromSmiles(fragment), depth=3)
            fragment = random.choice(cores)
            
        encoded_fragment = sf.SAFEConverter(ignore_stereo=True).encoder(fragment, allow_empty=True) + '.'
        x = self.model.tokenizer([encoded_fragment],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples)
        samples = self.generate(x, softmax_temp, randomness, gamma=gamma)

        if apply_filter:
            return filter_by_substructure(samples, fragment)
        return samples

    @staticmethod
    def _as_protected_list(protected_smiles):
        """Normalize protected_smiles to a list of SMILES, or None."""
        if protected_smiles is None:
            return None
        if isinstance(protected_smiles, str):
            return [protected_smiles]
        return [s for s in protected_smiles if s]

    def _safe_fragment_indices_matching(self, smiles, protected_smiles):
        """Return SAFE fragment indices overlapping any entry in *protected_smiles*.

        *protected_smiles* may be a single SMILES string or a list of SMILES.
        """
        encoded = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(
            smiles, allow_empty=True,
        )
        protected_list = self._as_protected_list(protected_smiles)
        if not protected_list:
            return set()
        protected_mols = []
        for smi in protected_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                protected_mols.append(mol)
        if not protected_mols:
            return set()
        protected_idx = set()
        for i, frag in enumerate(encoded.split('.')):
            frag_smi = sf.decode(frag, canonical=True, ignore_errors=True)
            if frag_smi is None:
                continue
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol is None:
                continue
            for protected in protected_mols:
                if (
                    frag_mol.HasSubstructMatch(protected)
                    or protected.HasSubstructMatch(frag_mol)
                ):
                    protected_idx.add(i)
                    break
        return protected_idx

    def _map_safe_fragments_to_atoms(self, smiles, mol=None):
        """Map each SAFE fragment to heavy-atom indices in *mol* (greedy, unused-first).

        Returns a list of tuples (one per SAFE fragment). Atoms are indices into
        *mol*, which defaults to ``Chem.MolFromSmiles(smiles)``.
        """
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES for fragment mapping: {smiles}')
        encoded = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(
            smiles, allow_empty=True,
        )
        assigned = set()
        mapping = []
        for frag in encoded.split('.'):
            frag_smi = sf.decode(frag, canonical=True, ignore_errors=True)
            if frag_smi is None:
                mapping.append(())
                continue
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol is None:
                mapping.append(())
                continue
            matches = mol.GetSubstructMatches(frag_mol)
            best = None
            for match in matches:
                if not set(match) & assigned:
                    best = match
                    break
            if best is None and matches:
                best = matches[0]
            if best is None:
                mapping.append(())
                continue
            assigned.update(best)
            mapping.append(tuple(best))
        return mapping

    def _safe_fragment_indices_overlapping_atoms(self, smiles, atom_indices, mol=None):
        """Return SAFE fragment indices that overlap any of *atom_indices*."""
        target = set(int(i) for i in atom_indices)
        if not target:
            return set()
        mapping = self._map_safe_fragments_to_atoms(smiles, mol=mol)
        return {i for i, atoms in enumerate(mapping) if target & set(atoms)}

    def mask_modification(self, smiles, min_len=30, protected_smiles=None, **kwargs):
        encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
        x = self.model.tokenizer([encoded_smiles],
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=self.model.config.model.max_position_embeddings)['input_ids']
        if protected_smiles is not None:
            kwargs['protected_smiles'] = protected_smiles
        if x.shape[-1] < min_len:
            return self.addmask(smiles, num_edit=min_len-x.shape[-1]+1, **kwargs)
        return self.remask(smiles, input_ids=x, **kwargs)

    def addmask(self, smiles, num_edit=3, protected_smiles=None, **kwargs):
        try:
            samples = self.fragment_completion(smiles, mask_len=num_edit, apply_filter=False, **kwargs)
        except:
            return smiles
        protected_list = self._as_protected_list(protected_smiles)
        if protected_list and samples:
            for prot in protected_list:
                samples = filter_by_substructure(samples, prot)
                if not samples:
                    break
        if samples:
            return samples[0]
        return smiles
    
    def remask(self, smiles, input_ids=None, protected_smiles=None,
               mask_frag_indices=None, mask_all=False, **kwargs):
        """Remask one or more SAFE fragments and decode.

        Args:
            smiles: Input molecule SMILES.
            input_ids: Optional pre-tokenized SAFE sequence.
            protected_smiles: Fragment(s) that must not be remasked. Ignored when
                *mask_frag_indices* is set.
            mask_frag_indices: Explicit SAFE fragment indices to remask (e.g. from
                substructure inpainting). When set, only these fragments are candidates.
            mask_all: If True and multiple candidates exist, remask all of them in
                one pass (FLOWR-style region inpainting). If False, remask one
                randomly chosen candidate.
        """
        x = input_ids
        if x is None:
            encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
            x = self.model.tokenizer([encoded_smiles],
                                     return_tensors='pt',
                                     truncation=True,
                                     max_length=self.model.config.model.max_position_embeddings)['input_ids']
        
        # fragment mask replacement
        special_token_idx = [0] + (x[0] == self.dot_index).nonzero(as_tuple=True)[0].tolist() + [len(x[0]) - 1]
        n_frags = len(special_token_idx) - 1
        if mask_frag_indices is not None:
            candidates = sorted({int(i) for i in mask_frag_indices if 0 <= int(i) < n_frags})
            if not candidates:
                return smiles
        elif protected_smiles is not None:
            protected_idx = self._safe_fragment_indices_matching(smiles, protected_smiles)
            candidates = [i for i in range(n_frags) if i not in protected_idx]
            if not candidates:
                return smiles
        else:
            candidates = list(range(n_frags))
            if not candidates:
                return smiles

        if mask_all and len(candidates) > 1:
            frags_to_mask = sorted(candidates, reverse=True)
        else:
            frags_to_mask = [random.choice(candidates)]

        for frag_idx in frags_to_mask:
            special_token_idx = (
                [0]
                + (x[0] == self.dot_index).nonzero(as_tuple=True)[0].tolist()
                + [len(x[0]) - 1]
            )
            if frag_idx + 1 >= len(special_token_idx):
                continue
            mask_start_idx = special_token_idx[frag_idx] + 1
            mask_end_idx = special_token_idx[frag_idx + 1]
            num_insert_mask = random.randint(5, 15)
            num_insert_mask = min(
                num_insert_mask,
                self.model.config.model.max_position_embeddings - x.shape[-1]
                + mask_end_idx - mask_start_idx,
            )
            if num_insert_mask < 1:
                continue
            x = torch.hstack([
                x[:, :mask_start_idx],
                torch.full((1, num_insert_mask), self.model.mask_index),
                x[:, mask_end_idx:],
            ])
        samples = self.generate(x, **kwargs)
        if samples:
            return samples[0]
        return smiles

    def inpaint(self, smiles, mask_atom_indices=None, mol=None,
                num_samples=1, mask_all=True, keep_smiles=None, **kwargs):
        """Inpaint SAFE fragments that overlap *mask_atom_indices*.

        Args:
            smiles: Molecule SMILES used for SAFE encoding / generation.
            mask_atom_indices: Heavy-atom indices (into *mol* or MolFromSmiles(smiles))
                that should be changed — FLOWR ``--substructure`` semantics.
            mol: Optional RDKit mol whose atom indices match *mask_atom_indices*.
            num_samples: Number of independent remask draws.
            mask_all: Remask all overlapping SAFE fragments in one pass.
            keep_smiles: Optional SMILES / list used to filter completions that
                still contain the kept region (``--filter_cond_substructure``).
        """
        if not mask_atom_indices:
            raise ValueError('mask_atom_indices must be a non-empty list of atom indices')
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        frag_idx = self._safe_fragment_indices_overlapping_atoms(
            smiles, mask_atom_indices, mol=mol,
        )
        if not frag_idx:
            raise ValueError(
                'No SAFE fragment overlaps the requested mask atoms; '
                'check --substructure indices against the heavy-atom molecule.'
            )
        keep_list = self._as_protected_list(keep_smiles)
        samples = []
        for _ in range(num_samples):
            sample = self.remask(
                smiles,
                mask_frag_indices=frag_idx,
                mask_all=mask_all,
                **kwargs,
            )
            if not sample or sample == smiles:
                continue
            if keep_list:
                filtered = [sample]
                for prot in keep_list:
                    filtered = filter_by_substructure(filtered, prot)
                    if not filtered:
                        break
                if not filtered:
                    continue
                sample = filtered[0]
            samples.append(sample)
        return samples
