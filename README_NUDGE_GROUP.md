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
   Python 3.12.3
   ```
   Make sure your Python version **is not newer than 3.12**. Otherwise it leads to a dependency conflict for torch, as python 3.13 uses torch==2.6 instead of torch==2.4, which was used in the project.

## Set Up 

### Set Up new Virtual Environment
   ```
   python3 -m venv env
   . env/bin/activate
   ```

If you see an error message like this:

> *The virtual environment was not created successfully because ensurepip is not available.  
> On Debian/Ubuntu systems, you need to install the python3-venv package...*

It means the `python3.12-venv` package is missing.

To fix this, run the following commands:

```
   sudo apt update
   sudo apt install python<your python version>-venv
   ```

Repeat the original two commands.

### Set Up the Repository

1. Clone the repository: 
   ```bash
   git clone -b Nudge https://github.com/ml-research/blendrl NudgeTest
   ```
   *Change NudgeTest to whatever you like or omit*

2. Open the cloned repository.
   ```bash
   cd NudgeTest
   ```
   *Same Folder Name as in 1., blendrl if omitted*


4. Install all requirements via
   ```bash
   pip install -r requirements.txt
   ```

2. Install other dependencies
    ```bash
    cd nsfr
    pip install -e .
    cd ../nudge
    pip install -e .
    cd ../neumann
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


   Check what version of PyTorch you are on (CPU or GPU):
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
   ```

   You should get something like this if you are using GPU version.
   ```bash
   2.4.0+cu121
   12.1
   ```

   If you get something like this it means you are currently using CPU:

   ```bash
   2.4.0+cpu
   None
   ```


   Based on the output above use the respective commands.

   For CPU:
   ```bash
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
   pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
   ```

   For CUDA 12.1:
   ```bash
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   ```
   *Change CUDA version if necessary. If you have issues please follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).*

   *Make sure you're using the correct torch and torch-scatter/torch-sparse version for CPU or CUDA. The CPU version of torch is not compatible with CUDA torch-scatter/torch-sparse versions. **Mismatched versions will cause training to fail.***


## Run trained agents

Use the following commands to run the agents we trained on the GPU

1. **Freeway:**
   ```bash
   python play_gui.py --env-name freeway --agent-path out_freeway/runs/freeway_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0
   ```

2. **Space Invaders:**
   ```bash
   python play_gui.py --env-name spaceinvaders --agent-path  out_spaceinvaders/runs/spaceinvaders_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0
   ```


3. **Breakout:**
   ```bash
   python play_gui.py --env-name breakout --agent-path out_breakout/runs/breakout_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_100_steps_128__0
   ```

4. **Kangaroo**
   ```
   python play_gui.py --env-name kangaroo --agent-path out_kangaroo/runs/kangaroo_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0
   ```

5. **Pong:**
   ```bash
   python play_gui.py --env-name pong --agent-path out_pong/runs/pong_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_100_steps_128__0
   ```

6. **Seaquest:**
   ```bash
   python play_gui.py --env-name seaquest --agent-path out_seaquest/runs/seaquest_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0
   ```

7. **Donkeykong:**
   ```bash
   python play_gui.py --env-name donkeykong --agent-path out_donkeykong/runs/donkeykong_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_100_steps_128__0
   ```
*Attention. The weights for Donekykong are not updating correctly because of the unidentified technical problems. During the creation of the environment and testing the problem was not present. More information is provided in the Project Report.*


## Train agents
You have to use different commands to start training an agent.

To explicit train NUDGE agent use flag `--actor_mode logic` which activates only logic part of BlendRL.

Do not forget to change `--num-envs` parameter. 
- For local testing we suggest using **5-10** environments. 
- For real training with GPU use **500** environments. If it is not possible due to technical constraints use **50-100** environments.

To briefly test that training runs on all Environments use the following commands:

*If you want to complete a quick training run instead of aborting it, add `--total-timesteps 5000 --save-steps 1000` to the following commands (feel free to adjust the numbers, make sure total-timesteps is greater than save-steps)*
1. **Freeway:**
   ```
   python train_blenderl.py --env-name freeway --joint-training --num-envs 5 --actor_mode logic
   ```

2. **Space Invaders:**
   ```bash
   python train_blenderl.py --env-name spaceinvaders --joint-training --num-envs 5 --actor_mode logic
   ```


3. **Breakout:**
   ```bash
   python train_blenderl.py --env-name breakout --joint-training --num-envs 5 --actor_mode logic
   ```

4. **Kangaroo:**
   ```bash
   python train_blenderl.py --env-name kangaroo --joint-training --num-envs 5 --actor_mode logic
   ```

5. **Pong:**
   ```bash
   python train_blenderl.py --env-name pong --joint-training --num-envs 5 --actor_mode logic
   ```

6. **Seaquest:**
   ```bash
   python train_blenderl.py --env-name seaquest --joint-training --num-envs 5 --actor_mode logic
   ```

7. **Donkeykong:**
   ```bash
   python train_blenderl.py --env-name donkeykong --joint-training --num-envs 5 --actor_mode logic
   ```

The following commands were used to train each respective agent. The needed flags and number of environments are already set.


1. **Freeway:**
   ```bash
   python train_blenderl.py --env-name freeway --joint-training --num-steps 128 --num-envs 50 --gamma 0.99 --actor_mode logic
   ```

2. **Space Invaders:**
   ```bash
   python train_blenderl.py --env-name spaceinvaders --joint-training --num-steps 128 --num-envs 50 --gamma 0.99 --actor_mode logic
   ```


3. **Breakout:**
   ```bash
   python train_blenderl.py --env-name breakout --joint-training --num-steps 128 --num-envs 100 --gamma 0.99 --actor_mode logic
   ```

4. **Kangaroo:**
   ```bash
   python train_blenderl.py --env-name kangaroo --joint-training --num-steps 128 --num-envs 50 --gamma 0.99 --actor_mode logic
   ```

5. **Pong:**
   ```bash
   python train_blenderl.py --env-name pong --joint-training --num-steps 128 --num-envs 100 --gamma 0.99 --actor_mode logic
   ```

6. **Seaquest:**
   ```bash
   python train_blenderl.py --env-name seaquest --joint-training --num-steps 128 --num-envs 50 --gamma 0.99 --actor_mode logic
   ```

7. **Donkeykong:**
   ```bash
   python train_blenderl.py --env-name donkeykong --joint-training --num-steps 128 --num-envs 100 --gamma 0.99 --actor_mode logic
   ```