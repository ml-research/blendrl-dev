from pathlib import Path

import jinja2 as j2
import tyro
import yaml
from dataclasses import dataclass


@dataclass
class Args:
    env_name: str
    """Name of the environment."""
    prompt_name: str
    """Name of the prompt."""

def main():
    args = tyro.cli(Args)

    base_dir = Path("in/envs") / args.env_name / "priors"
    config_path = base_dir / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    prompt_path = Path("in/prompts") / (args.prompt_name + ".j2")
    with open(prompt_path, "r") as f:
        template = j2.Template(f.read())

    for predicate in config["predicates"]:
        print("=" * 80)
        print(predicate["name"])
        print("=" * 80)
        config["predicate"] = predicate
        print(template.render(**config))
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()