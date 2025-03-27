# Installation

1. Install all requirements via
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
    ```

3. Optional: Install NEUMANN dependencies for memory-efficient reasoning. This will be required only when the nsfr reasoner produces an out-of-memory error due to highly-parallelized environment, e.g. 512 environments in Seaquest.
    ```bash
    cd neumann
    pip install -e .
    ```
    It requires [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and its dependencies.
    ```bash
    pip install torch-geometric
    pip install torch-sparse
    pip install torch-scatter
    ```
    For Mac users, you may need to install `torch-geometric` from source. 
    ```
    git clone https://github.com/rusty1s/pytorch_scatter.git
    cd pytorch_scatter
    python setup.py install
    ```
    To utilize GPUs, you would need to install `torch-geometric` with CUDA support, e.g. with CUDA 12.4.
    ```bash
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
    ``` 
    You may need to [update torch to match your CUDA version](https://pytorch.org/get-started/previous-versions/).
    ```
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    ```
Otherwise, please refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

4. You can now run the training script, e.g.:
    ```bash
    python train_blenderl.py --env-name seaquest --joint-training --num-steps 128 --num-envs 5 --gamma 0.99　
    ```
    Scripts of the full training to reproduce the result are provided [here](scripts/training.sh).
<!-- 
5. You can also run the evaluation script:
    ```bash
    python evaluate.py --env-name seaquest --agent-path models/seaquest_demo
    ``` -->
