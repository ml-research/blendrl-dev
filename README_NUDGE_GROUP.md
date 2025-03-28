# How to Test and Use the NUDGE Part inside the BlendRL Repository

Inna Kuzmina, Niklas Kuchenbrandt, Dominic Eib, Tim Klein

## Verify your Python Version

1. Open the Console / Terminal
    - **Windows**: Press `Win + R`, type `cmd`, and press `Enter`
    - **macOS**: Open **Terminal** via Spotlight (`Cmd + Space`, then type `Terminal`)
    - **Linux**: Open your **Terminal** (e.g., `Ctrl + Alt + T`)


2. Run one of the following commands:
 ```bash
   python --version 
```
If that doesn’t work, try:

```bash
   python3 --version
```

You should see something like:
```bash
   Python 3.12.5
```
Make sure you use evrsion **not older than 3.12**. Othervise it leads to a dependency conflict for the torch, as python 3.13 uses torch==2.6 instead of torch==2.4 used in the project.


## Set Up the Repository

1. Clone the repository: https://github.com/ml-research/blendrl/tree/Nudge

2. Open the cloned repository in your code editor.

3. Switch to the **Nudge** Branch
```bash
   git checkout -b Nudge origin/Nudge
```

4. Install all requirements via
```bash
   pip install -r requirements.txt
```

2. Install other dependencies
    ```bash
    cd nsfr
    pip install -e .
    cd ..
    cd nudge
    pip install -e .
    cd ..
    cd neumann
    pip install -e .
    cd ..
    ```
3. Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)  dependencies required by neumann.
```bash
   pip install torch-geometric
   pip install torch-sparse
   pip install torch-scatter
   ```
If you have problems with installing **torch-sparse and torch-scatter** use the respective commands below. 

If you are using CPU:
```bash
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
   pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
   ```

For CUDA 12.1:
```bash
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```
*Change CUDA version if necessary.*

## Run the trained agents

You have to use different commands to run trained agents for a specific game.

1. To run **freeway** use:
```bash
   python play_gui.py --env-name freeway --agent-path out_freeway\runs\freeway_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0
```

2. To run **space invaders** use:
```bash
   python play_gui.py --env-name spaceinvaders --agent-path  out_spaceinvaders\runs\spaceinvaders_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0
```


3. To run **breakout** use:
```bash
   python play_gui.py --env-name breakout --agent-path out_breakout/runs/breakout_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_100_steps_128__0
```

4. To run **kangaroo** use:
```bash
   python play_gui.py --env-name kangaroo --agent-path out_kangaroo/runs/kangaroo_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0
```

5. To run **pong** use:
```bash
   python play_gui.py --env-name pong --agent-path out_pong/runs/pong_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_100_steps_128__0
```

6. To run **seaquest** use:
```bash
   python play_gui.py --env-name seaquest --agent-path out_seaquest/runs/seaquest_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0
```

7. To run **donkeykong** use:
```bash
   python play_gui.py --env-name donkeykong --agent-path out_dk/runs/donkeykong_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_100_steps_128__0
```
*Attention. The weights are not updating correctly because of the unidentified technical  problems. During the creation of the environment and testing the problem was not present.*


## Train your own agents
You have to use different commands to start training an agent.

To explicit train NUDGE agent use flag `--actor_mode logic` which activates only logic part of BlendRL.

Do not forget to change `--num-envs` parameter. 
- For local testing we suggest using **5-10** environments. 
- For real training with GPU use **500** environments. If it is not possible due to technical constraints use **50-100** environments.

The following commands were used to train each respective agent. The needed flags and number of environments are already set.


1. To train **freeway** use:
```bash
   python train_blenderl.py --env-name freeway --joint-training --num-steps 128 --num-envs 50 --gamma 0.99 --actor_mode logic
```

2. To train **space invaders** use:
```bash
   python train_blenderl.py --env-name spaceinvaders --joint-training --num-steps 128 --num-envs 50 --gamma 0.99 --actor_mode logic
```


3. To train **breakout** use:
```bash
   python train_blenderl.py --env-name breakout --joint-training --num-steps 128 --num-envs 100 --gamma 0.99 --actor_mode logic
```

4. To train **kangaroo** use:
```bash
   python train_blenderl.py --env-name kangaroo --joint-training --num-steps 128 --num-envs 50 --gamma 0.99 --actor_mode logic
```

5. To train **pong** use:
```bash
   python train_blenderl.py --env-name pong --joint-training --num-steps 128 --num-envs 100 --gamma 0.99 --actor_mode logic
```

6. To train **seaquest** use:
```bash
   python train_blenderl.py --env-name seaquest --joint-training --num-steps 128 --num-envs 50 --gamma 0.99 --actor_mode logic
```

7. To train **donkeykong** use:
```bash
   python train_blenderl.py --env-name seaquest --joint-training --num-steps 128 --num-envs 100 --gamma 0.99 --actor_mode logic
```


## How to reproduce our analysis results

You have to use `analysis_with randomb_movingavg.py` file in the root dierection.

Two important changes to do in the script:

1. Line 9 change the path to the trained agent. You can use the path from the instructions how to run agents. 
2. Line 11 change the name of the environment (e.g. `env_name="ALE/Pong-v5"`) you want to run the radom baseline for. 

Install tensorflow:
```bash
   pip install tensorflow
```

Run the python script. 
```bash
   python analysis_with randomb_movingavg.py
```
