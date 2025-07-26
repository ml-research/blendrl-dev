from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from nsfr.fol.logic import Clause, Atom
from nsfr.utils.logic import get_index_by_predname


class NSFReasoner(nn.Module):
    """The Neuro-Symbolic Forward Reasoner.

    Args:
        perception_model (nn.Module): The perception model.
        facts_converter (nn.Module): The facts converter module.
        infer_module (nn.Module): The differentiable forward-chaining inference module.
        atoms (list(atom)): The set of ground atoms (facts).
    """

    def __init__(self, facts_converter, infer_module, atoms: List[Atom], bk, clauses: List[Clause], device, train=False, explain=False):
        super().__init__()
        self.fc = facts_converter
        self.im = infer_module
        self.atoms = atoms
        self.bk = bk
        self.clauses = clauses
        self.device = device
        self._train = train
        self.explain = explain
        self.prednames = self.get_prednames()
        self.V_0 = []
        self.V_T = []

    def get_params(self):
        # Include predicate model parameters along with infer_module parameters
        params = self.im.get_params()
        if self.predicate_model is not None:
            params += list(self.predicate_model.parameters())
        return params

    def get_prednames(self):
        prednames = []
        for clause in self.clauses:
            if clause.head.pred.name not in prednames:
                prednames.append(clause.head.pred.name)
        return prednames

    def compute_values(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        V_0 = self.init_values(x)
        V_T = self.im(V_0)

        return V_0, V_T

    def forward(self, x):
        self.V_0, self.V_T = self.compute_values(x)

        # only return probs of actions
        actions = self.get_predictions(self.V_T, prednames=self.prednames)
        return actions

    def init_values(self, x: torch.Tensor) -> torch.Tensor:
        # convert to the valuation tensor
        V_0 = self.fc(x, self.atoms, self.bk)

        # dummy variable to compute input gradients
        if self.explain:
            self.dummy_zeros = torch.zeros_like(V_0, requires_grad=True).to(torch.float32).to(self.device)
            self.dummy_zeros.requires_grad_()
            self.dummy_zeros.retain_grad()
            # add dummy zeros to get input gradients
            V_0 = V_0 + self.dummy_zeros

        return V_0

    def predict(self, v, predname):
        """Extract a value from the valuation tensor using a given predicate."""
        # v: batch * |atoms|
        target_index = get_index_by_predname(
            pred_str=predname, atoms=self.atoms)
        return v[:, target_index]

    def predict_multi(self, v, prednames):
        """Extract values from the valuation tensor using given predicates."""
        # v: batch * |atoms|
        target_indices = []
        for predname in prednames:
            target_index = get_index_by_predname(
                pred_str=predname, atoms=self.atoms)
            target_indices.append(target_index)
        prob = torch.cat([v[:, i].unsqueeze(-1)
                          for i in target_indices], dim=1)
        B = v.size(0)
        N = len(prednames)
        assert prob.size(0) == B and prob.size(
            1) == N, 'Invalid shape in the prediction.'
        return prob

    def print_program(self):
        """Print a summary of logic programs using continuous weights."""
        # print('====== LEARNED PROGRAM ======')
        C = self.clauses
        # a = self.im.W
        Ws_softmaxed = torch.softmax(self.im.W, 1)
        
        # print("Raw rule weights: ")
        # print(self.im.W)
        # print("Softmaxed rule weights: ")
        # print(Ws_softmaxed)

        # print("Summary: ")
        for i, W_ in enumerate(Ws_softmaxed):
            max_i = np.argmax(W_.detach().cpu().numpy())
            print('C_' + str(i) + ': ',
                  C[max_i], 'W_' + str(i) + ':', round(W_[max_i].detach().cpu().item(), 3))

    def print_valuations(self, predicate: str = None, min_value: float = 0,
                         initial_valuation: bool = False):
        print('===== VALUATIONS =====')
        valuation = self.V_0 if initial_valuation else self.V_T
        for b, batch in enumerate(valuation):
            print(f"== BATCH {b} ==")
            batch = batch.detach().cpu().numpy()
            idxs = np.argsort(-batch)  # Sort by valuation value
            for i in idxs:
                value = batch[i]
                if value >= min_value:
                    atom = self.atoms[i]
                    if predicate is None or predicate == atom.pred.name:
                        print(f"{value:.3f} {atom}")
                        
    def print_valuations_input(self, V, predicate: str = None, min_value: float = 0):
        print('===== VALUATIONS =====')
        valuation = V
        for b, batch in enumerate(valuation):
            print(f"== BATCH {b} ==")
            batch = batch.detach().cpu().numpy()
            idxs = np.argsort(-batch)  # Sort by valuation value
            for i in idxs:
                value = batch[i]
                if value >= min_value:
                    atom = self.atoms[i]
                    if predicate is None or predicate == atom.pred.name:
                        print(f"{value:.3f} {atom}")

    def print_action_predicate_valuations(self, initial_valuation: bool = True):
        for predicate in self.prednames:
            self.print_valuation_for_predname(predicate, initial_valuation=initial_valuation)

    def print_valuation_for_predname(self, predname: str, initial_valuation: bool = True):
        value = self.get_predicate_valuation(predname, initial_valuation)
        print(f"{predname}:  {value:.3f}")

    def get_predicate_valuation(self, predname: str, initial_valuation: bool = True):
        valuation = self.V_0 if initial_valuation else self.V_T
        target_index = get_index_by_predname(pred_str=predname, atoms=self.atoms)
        value = valuation[:, target_index].item()
        return value
    
    def get_fact_valuation(self, predname: str, initial_valuation: bool = True):
        valuation = self.V_0 if initial_valuation else self.V_T
        target_index = get_index_by_predname(pred_str=predname, atoms=self.atoms)
        value = valuation[:, target_index].item()
        return value

    def print_explaining(self, predicts):
        predicts = predicts.detach().cpu().numpy()
        index = np.argmax(predicts[0])
        return self.prednames[index]

    def print_probs(self):
        probs = self.get_probs()
        for atom, p in probs.items():
            print(f"{p:.3f} {atom}")

    def get_probs(self, predname: Optional[str] = None, index: int = 0):
        probs = {}
        for i, atom in enumerate(self.atoms):
            if predname is None or predname == atom.pred.name:
                probs[atom] = self.V_T[index][i].item()
        return probs

    def get_valuation_text(self, valuation):
        text_batch = ''  # texts for each batch
        for b in range(valuation.size(0)):
            top_atoms = self.get_top_atoms(valuation[b].detach().cpu().numpy())
            text = '----BATCH ' + str(b) + '----\n'
            text += self.atoms_to_text(top_atoms)
            text += '\n'
            # texts.append(text)
            text_batch += text
        return text_batch

    def get_top_atoms(self, v):
        top_atoms = []
        for i, atom in enumerate(self.atoms):
            if v[i] > 0.7:
                top_atoms.append(atom)
        return top_atoms

    def atoms_to_text(self, atoms):
        text = ''
        for atom in atoms:
            text += str(atom) + ', '
        return text

    def get_predictions(self, V_T, prednames):
        predicts = self.predict_multi(v=V_T, prednames=prednames)
        return predicts
