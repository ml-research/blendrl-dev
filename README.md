### *Refactoring is undergoing.
# BlendRL: A Framework for Merging Symbolic and Neural Policies (ICLR 2025)
[Hikaru Shindo](https://www.hikarushindo.com/), [Quentin Delfosse](https://ml-research.github.io/people/qdelfosse/index.html), [Devendra Singh Dhami](https://sites.google.com/view/devendradhami), [Kristian Kersting](https://ml-research.github.io/people/kkersting/index.html)


*We propose a framework that jointly learns symbolic and neural policies for reinforcement learning.*
<!-- <img src="assets/blenderl.png" alt="drawing" height="200"/> -->
<img src="assets/seaquest_agent.gif" width=800>
<img src="assets/kangaroo_agent.gif" width=800>

## Quickstart

### Installation
Follow [INSTALLATION.md](INSTALLATION.md) to install dependencies.


<!--
1. Install all requirements via
    ```bash
    pip install -r requirements.txt
    ```
2. On project level, simply run `python train.py` to start a new training run.
-->

Download the trained agents:
```
wget https://hessenbox.tu-darmstadt.de/dl/fiCNznPuWkALH8JaCJWHeeAV/models.zip
unzip models.zip
rm models.zip
```
Then you can run the play script:
```
python play_gui.py --env-name kangaroo --agent-path models/kangaroo_demo
python play_gui.py --env-name seaquest --agent-path models/seaquest_demo
```
Note that a checkpoint is required to run the play script.


<!-- You can run the training script:
```
python train_blenderl.py --env-name seaquest --joint-training --num-steps 128 --num-envs 5 --gamma 0.99
```
- --joint-training: train neural and logic modules jointly
- --num-steps: the number of steps for policy rollout
- --num-envs: the number of environments to train agents
- --gamma: the discount factor for future rewards -->

**Train the valuation functions:**
```
python train_valuation.py --env-name kangaroo --num-steps 128 --num-envs 5 --track
```

- --num-steps: the number of steps for policy rollout
- --num-envs: the number of environments to train agents
- --gamma: the discount factor for future rewards
- --track: track the training process with wandb


## How to Use
<!-- ### Hyperparameters
The hyperparameters are configured inside `in/config/default.yaml` which is loaded as default. You can specify a different configuration by providing the corresponding YAML file path as an argument, e.g., `python train.py in/config/my_config.yaml`. A description of all hyperparameters can be found in `train.py`. -->

### The Logic
Inside `in/envs/[env_name]/logic/[ruleset_name]/`, you find the logic rules that are used as a starting point for training. You can change them or create new rule sets. The ruleset to use is specified with the hyperparam `rules`.

<!-- ### Install Locally
If you want to use NUDGE within other projects, you can install NUDGE locally as follows:
1. Inside ```nsfr/``` run
    ```bash
    python setup.py develop
    ```
2. Inside ```nudge/``` run
    ```bash
    python setup.py develop
    ``` -->

<!-- ### Ohter dependencies
1. Install packages by `pip install -r requirements.txt` 

2. PyG and torch-scatter for neumann
Install PyG and torch-scatter packages for neumann reasoner. See the [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). These should be consistent in terms of ther versions, e.g.
    ```
    pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html 
    ``` -->


## How to Set up New Environments
You add a new environment inside `in/envs/[new_env_name]/`. There, you need to define a `NudgeEnv` class that wraps the original environment in order to do
* **logic state extraction**: translates raw env states into logic representations
* **valuation**: Each relation (like `closeby`) has a corresponding **valuation function** which maps the (logic) game state to a probability that the relation is true. Each valuation function is defined as a simple Python function. The function's name must match the name of the corresponding relation.
* **action mapping**: action-predicates predicted by the agent need to be mapped to the actual env actions

See the `freeway` env to see how it is done.


# Plots

**Preparation:**
```bash
mkdir plots
mkdir plots/predicate_valuation
mkdir plots/curves
```

**Kangaroo 1st stage:** (Fig. 5.1)
```bash
python3 -m plot.plot_predicate_valuation \
  --exp-name original_kangaroo claude_kangaroo chatgpt_kangaroo mlp_257_cl_cc0_3_s1 mlp_257_gpt_cc0_3_s1 \
  --predicate-name left_of_ladder right_of_ladder on_ladder \
  --position Ladder_1 Ladder_2 Ladder_3 \
  --plots valuation_overlay \
  --contour-objects Ladder Platform \
  --save-grid \
  --grid-labels "BlendRL" "Claude" "ChatGPT" "Ours (Claude)" "Ours (ChatGPT)" \
  --resolution 256 \
  --value-colorbar-index 0 \
  --grid-filename "kangaroo_1st" \
  --grid-title "Kangaroo"
```

**Seaquest 1st stage:** (Fig. 5.2)
```bash
python3 -m plot.plot_predicate_valuation \
  --exp-name original_seaquest claude_seaquest chatgpt_seaquest mlp_357_cl_cc0_1_s0 mlp_357_gpt_cc0_3_s1 \
  --predicate-name left_of_diver right_of_diver higher_than_diver deeper_than_diver \
  --position 5 \
  --plots valuation_overlay \
  --save-grid \
  --grid-labels "BlendRL" "Claude" "ChatGPT" "Ours (Claude)" "Ours (ChatGPT)" \
  --resolution 256 \
  --value-colorbar-index 0 \
  --grid-filename "seaquest_1st" \
  --grid-title "Seaquest"
```

**Neural blending weights:** (Fig. 5.3)

Kangaroo (ep0):
```bash
python3 -m plot.plot_simulation \
  --sim-name test_hybrid \
  --exp-name mlp_216_gpt_def_s513
```

Seaquest (ep1):
```bash
python3 -m plot.plot_simulation \
  --sim-name test_hybrid \
  --exp-name mlp_316_gpt_def_s1024
```

**Kangaroo 2nd stage:** (Fig. 5.5)
```bash
python3 -m plot.plot_predicate_valuation \
  --exp-name original_kangaroo claude_kangaroo chatgpt_kangaroo mlp_216_cl_def_s515 mlp_216_gpt_def_s513 \
  --predicate-name close_by_monkey close_by_throwncoconut \
  --position 5 \
  --plots valuation_overlay \
  --contour-objects Ladder Platform \
  --save-grid \
  --grid-labels "BlendRL" "Claude" "ChatGPT" "Ours (Claude)" "Ours (ChatGPT)" \
  --resolution 256 \
  --value-colorbar-index 0 \
  --grid-filename "kangaroo_2nd" \
  --grid-title "Kangaroo"
```

**Seaquest 2nd stage:** (Fig. 5.6)
```bash
python3 -m plot.plot_predicate_valuation \
  --exp-name original_seaquest claude_seaquest chatgpt_seaquest mlp_316_cl_def_s1024 mlp_316_gpt_def_s1024 \
  --predicate-name close_by_enemy close_by_missile \
  --position 5 \
  --plots valuation_overlay \
  --save-grid \
  --grid-labels "BlendRL" "Claude" "ChatGPT" "Ours (Claude)" "Ours (ChatGPT)" \
  --resolution 256 \
  --value-colorbar-index 0 \
  --grid-filename "seaquest_2nd" \
  --grid-title "Seaquest"
```

**(Ablation) Kangaroo without concept aligner:** (Fig. 5.7)
```bash
python3 -m plot.plot_predicate_valuation \
  --exp-name mlp_213_s513 mlp_213_s514 mlp_213_s515 \
  --predicate-name left_of_ladder right_of_ladder on_ladder \
  --position Ladder_1 Ladder_2 Ladder_3 \
  --plots valuation_overlay \
  --contour-objects Ladder Platform \
  --save-grid \
  --grid-labels "Seed 1" "Seed 2" "Seed 3" \
  --resolution 256 \
  --value-colorbar-index 0 \
  --grid-filename "kangaroo_logic_baseline_with_actions" \
  --grid-title 'Kangaroo ($c_{\operatorname{CA}}=0$)' \
  --grid-add-max-actions \
  --reweight \
  --action-clauses right_ladder left_ladder up_ladder
```

**(Ablation) Seaquest without concept aligner:** (Fig. 5.8)
```bash
python3 -m plot.plot_predicate_valuation \
  --exp-name mlp_313_s1024 mlp_313_s1025 mlp_313_s1026 \
  --predicate-name left_of_diver right_of_diver higher_than_diver deeper_than_diver \
  --position 5 \
  --plots valuation_overlay \
  --save-grid \
  --grid-labels "Seed 1" "Seed 2" "Seed 3" \
  --resolution 256 \
  --value-colorbar-index 0 \
  --grid-filename "seaquest_logic_baseline" \
  --grid-title 'Seaquest ($c_{\operatorname{CA}}=0$)'
```

**(Ablation) Kangaroo without annealing concept alignment loss:** (Fig. 5.9)
```bash
python3 -m plot.plot_predicate_valuation \
  --exp-name mlp_209_cl_cc0_03_s2 mlp_209_cl_cc0_1_s2 mlp_209_cl_cc0_3_s1 mlp_209_cl_cc1_0_s2 \
  --predicate-name left_of_ladder right_of_ladder on_ladder \
  --position Ladder_1 Ladder_2 Ladder_3 \
  --plots valuation_overlay \
  --contour-objects Ladder Platform \
  --save-grid \
  --grid-labels '$c_{\operatorname{CA}}=0.03$' '$c_{\operatorname{CA}}=0.1$' '$c_{\operatorname{CA}}=0.3$' '$c_{\operatorname{CA}}=1.0$' \
  --resolution 256 \
  --value-colorbar-index 0 \
  --grid-filename "kangaroo_claude_ca_woa" \
  --grid-title 'Kangaroo ($\gamma_{\operatorname{CA}}=0$)'
```

**(Ablation) Kangaroo with annealing concept alignment loss:** (Fig. 5.10)
```bash
python3 -m plot.plot_predicate_valuation \
  --exp-name mlp_209_cl_s2 mlp_257_cl_cc0_1_s2 mlp_257_cl_cc0_3_s1 mlp_257_cl_cc1_0_s2 \
  --predicate-name left_of_ladder right_of_ladder on_ladder \
  --position Ladder_1 Ladder_2 Ladder_3 \
  --plots valuation_overlay \
  --contour-objects Ladder Platform \
  --save-grid \
  --grid-labels '$c_{\operatorname{CA}}=0.03' '$c_{\operatorname{CA}}=0.1$' '$c_{\operatorname{CA}}=0.3$' '$c_{\operatorname{CA}}=1.0$' \
  --resolution 256 \
  --value-colorbar-index 0 \
  --grid-filename "kangaroo_claude_ca_wa" \
  --grid-title 'Kangaroo ($\gamma_{\operatorname{CA}}=1$)'
```

**(Ablation) Kangaroo concept alignment loss vs. performance:** (Fig. 5.11)

Requires access to the WandB project.
```bash
python3 -m plot.plot_loss_vs_performance \
  --entity-name roessler-thesis \
  --project-name blendRL_val_kangaroo
```

**Examples of concept misalignment:** (Fig 6.1)

Left of ladder:
```bash
python -m plot.plot_predicate_valuation \
  --exp-name mlp_209_cl_s0 mlp_209_cl_cc0_03_s2 \
  --predicate-name left_of_ladder \
  --position Ladder_1 Ladder_2 Ladder_3 \
  --contour-objects Ladder Platform \
  --plots valuation \
  --resolution 256
```

Right of ladder:
```bash
python -m plot.plot_predicate_valuation \
  --exp-name mlp_137_s0 \
  --predicate-name right_of_ladder \
  --position Ladder_1 Ladder_2 Ladder_3 \
  --contour-objects Ladder Platform \
  --plots valuation \
  --resolution 256
```

On ladder:
```bash
python -m plot.plot_predicate_valuation \
  --exp-name mlp_105_cl_s2 mlp_213_s513 \
  --predicate-name on_ladder \
  --position Ladder_1 Ladder_2 Ladder_3 \
  --contour-objects Ladder Platform \
  --plots valuation \
  --resolution 256
``