import os
from typing import Union

import torch

from nsfr.facts_converter import FactsConverter
from nsfr.nsfr import NSFReasoner
from nsfr.utils.logic import get_lang, get_blender_lang, build_infer_module
from nsfr.valuation import ValuationModule
from utils import optional


def get_nsfr_model(env_name: str, rules: str, device: Union[str, torch.device], train=False, explain=False, valuation_model=None, gamma: float = 0.01):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"in/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules)

    valuation_model = optional(valuation_model, f"in/envs/{env_name}/valuation.py")
    val_module = ValuationModule(valuation_model, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device, gamma=gamma)

    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(
        facts_converter=FC,
        infer_module=IM,
        atoms=atoms,
        bk=bk,
        clauses=clauses,
        device=device,
        train=train,
        explain=explain,
    )
    return NSFR


def get_blender_nsfr_model(env_name: str, rules: str, device: str, train=False, mode='normal', explain=False, valuation_model=None, gamma: float = 0.01):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"in/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_blender_lang(lark_path, lang_base_path, rules)

    valuation_model = optional(valuation_model, f"in/envs/{env_name}/valuation.py")
    val_module = ValuationModule(valuation_model, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(clauses)
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device, gamma=gamma)

    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(
        facts_converter=FC,
        infer_module=IM,
        atoms=atoms,
        bk=bk,
        clauses=clauses,
        device=device,
        train=train,
        explain=explain,
    )
    return NSFR