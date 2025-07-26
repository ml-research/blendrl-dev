import inspect
import re
from abc import ABC
from functools import partial
from typing import Dict, Union, Callable, List

import torch
from torch import nn

from nsfr.fol.language import Language
from nsfr.fol.logic import Atom, Const
from nsfr.utils.common import load_module


class ValuationModule(nn.Module, ABC):
    """Turns logic state representations into valuated atom probabilities according to
    the environment-specific valuation functions.

    Args:
        valuation_model: Either (a) the valuation model or (b) the path to the file containing the user-specified
            valuation functions.
    """

    lang: Language
    device: Union[torch.device, str]
    val_fns: Dict[str, Callable]  # predicate names to corresponding valuation fn

    def __init__(self, valuation_model: Union[nn.Module, str], lang: Language, device: Union[torch.device, str],
                 pretrained: bool = True):
        super().__init__()

        # Parse all valuation functions
        pred_names = set([pred.name for pred in lang.preds])
        if type(valuation_model) == str: # for backward compatibility
            val_fn_module = load_module(valuation_model)
            all_functions = inspect.getmembers(val_fn_module, inspect.isfunction)
            self.val_fns = {fn[0]: fn[1] for fn in all_functions if fn[0] in pred_names}
        else:
            self.val_fns = {pred_name: partial(valuation_model.forward, pred_name) for pred_name in pred_names}

        self.lang = lang
        self.device = device
        self.pretrained = pretrained

    def forward(self, Z: torch.Tensor, atom: Union[Atom, List[Atom]]):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                Z (tensor): The object-centric representation (the output of the YOLO model).
                atom (atom): The target atom to compute its probability.

            Returns:
                A batch of the probabilities of the target atom.
        """

        if isinstance(atom, Atom):
            return self.get_probs(Z, [atom])[0]
        else:
            return self.get_probs(Z, atom)

    def get_probs(self, Z: torch.Tensor, atoms: List[Atom]) -> torch.Tensor:
        batch_size = Z.shape[0]
        num_atoms = len(atoms)
        result = torch.zeros(batch_size, num_atoms, device=self.device)

        for i, atom in enumerate(atoms):
            result[:, i] = self._get_prob(Z, atom)

        return result

    def _get_prob(self, Z: torch.Tensor, atom: Atom) -> torch.Tensor:
        try:
            val_fn = self.val_fns[atom.pred.name]
        except KeyError as e:
            raise NotImplementedError(f"Missing implementation for valuation function '{atom.pred.name}'.")
        # term: logical term
        # args: the vectorized input evaluated by the value function
        args = [self.ground_to_tensor(term, Z) for term in atom.terms]
        return val_fn(*args)

    def ground_to_tensor(self, const: Const, zs: torch.Tensor):
        """Ground constant (term) into tensor representations.

            Args:
                const (const): The term to be grounded.
                zs (tensor): The object-centric state representation.
        """
        # Check if the constant name is in the reserved style, e.g., "obj1"
        result = re.match("obj([1-9][0-9]*)", const.name)
        if result is not None:
            # The constant is an object constant
            obj_id = result[1]
            obj_index = int(obj_id) - 1
            return zs[:, obj_index]

        elif const.dtype.name == 'object':
            obj_index = self.lang.term_index(const)
            return zs[:, obj_index]

        elif const.dtype.name == 'image':
            return zs

        else:
            return self.term_to_onehot(const, batch_size=zs.size(0))

    def term_to_onehot(self, term, batch_size):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        if term.dtype.name == 'color':
            return self.to_onehot_batch(self.colors.index(term.name), len(self.colors), batch_size)
        elif term.dtype.name == 'shape':
            return self.to_onehot_batch(self.shapes.index(term.name), len(self.shapes), batch_size)
        elif term.dtype.name == 'material':
            return self.to_onehot_batch(self.materials.index(term.name), len(self.materials), batch_size)
        elif term.dtype.name == 'size':
            return self.to_onehot_batch(self.sizes.index(term.name), len(self.sizes), batch_size)
        elif term.dtype.name == 'side':
            return self.to_onehot_batch(self.sides.index(term.name), len(self.sides), batch_size)
        elif term.dtype.name == 'type':
            return self.to_onehot_batch(self.lang.term_index(term), len(self.lang.get_by_dtype_name(term.dtype.name)),
                                        batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size."""
        onehot = torch.zeros(batch_size, length).to(self.device)
        onehot[:, i] = 1.0
        return onehot
