import os
import importlib.util
from nsfr.facts_converter import FactsConverter
from nsfr.utils.logic import get_lang, get_blender_lang, build_infer_module
from nsfr.nsfr import NSFReasoner
from nsfr.valuation import ValuationModule


def load_predicate_model(env_name: str, device: str):
    """Dynamically load the PredicateModel from the environment-specific valuation.py."""
    val_module_path = f"in/envs/{env_name}/valuation.py"
    spec = importlib.util.spec_from_file_location("valuation", val_module_path)
    valuation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(valuation)
    predicate_model = valuation.PredicateModel(device=device).to(device)
    return predicate_model


def get_nsfr_model(env_name: str, rules: str, device: str, train=False, explain=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"in/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules)

    val_fn_path = f"in/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    # Dynamically load the PredicateModel
    # predicate_model = load_predicate_model(env_name, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)

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
        # predicate_model=predicate_model
    )
    return NSFR


def get_blender_nsfr_model(env_name: str, rules: str, device: str, train=False, mode='normal', explain=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"in/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_blender_lang(lark_path, lang_base_path, rules)

    val_fn_path = f"in/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    # Dynamically load the PredicateModel
    # predicate_model = load_predicate_model(env_name, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(clauses)
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)

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
        # predicate_model=predicate_model
    )
    return NSFR