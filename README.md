# Source Code for A Bayesian-Symbolic Approach to Reasoning and Learning in Intuitive Physics

This repo contains the following folders and files.

`/`
- `notebooks/`: interactive notebooks to extract or analyse experimental results
    - `demo-world.ipynb`: example notebook to demonstrate how to run simulation with `BayesianSymbolic.jl`
    - `monly.ipynb`: notebook to extract results for M-step only results
    - `em.ipynb`: notebook to extract results for complete EM results
    - `phys101.ipynb`: notebook to extract results for PHYS101 results
    - `ullman.ipynb`: notebook to extract results for ULLMAN results
- `scripts/`: runnable scripts to generate or procepss data or to run experiments
    - `evalexpr_efficiency.jl`: script to compare different backends for expression evaluations; this is irrelavent to results
    - `generate-synth.jl`: script to generate SYNTH
    - `helper.jl`: common helper functions used by scripts
    - `neural.jl`: concrete architecture and hyper-parameters for neural baselines
    - `preprocess-ullman-master.jl`: script to process a scene from ULLMAN, calling `scripts/preprocess-ullman.py`
    - `preprocess-ullman.py`: script to process scenes from ULLMAN
    - `runexp_ullman.jl`: script to run an experiment on ULLMAN; for other datasets, use `runexp.jl`
    - `runexp.jl`: script to run an experiment on a given dataset and specific hyper-parameters; all datasets but ULLMAN are supported; for ULLMAN, use `runexp_ullman.jl`
    - `ullman_hacks.jl`: ULLMAN specific grammar (see the comments in the beginning of the file)
- `src/`: source codes for simulation and the BSP algorithm
    - `BayesianSymbolic.jl/`: world construction and simulations
    - `data/`: data processing functions
        - `phys101.jl`: PHYS101 specific functions
        - `preprocessing.jl`: generic data prepration functions
        - `ullman.jl`: ULLMAN specific functions
        - `ullman.txt`: ground truth information for ULLMAN
    - `scenarios/`: scenarios implemented in Turing.jl
        - `bounce.jl`: the BOUNCE scenario from SYNTH
        - `fall.jl`: the FALL scenario from PHYS101
        - `magnet.jl`: the MAGNET scenario; this is not used in the paper
        - `mat.jl`: the MAT scenario from SYNTH
        - `nbody.jl`: the NBDOY scenario from SYNTH
        - `spring.jl` the SPRING scenario from PHYS101
        - `ullman.jl`: the scenario from ULLMAN
    - `analyse.jl`: quantative and visual analysis
    - `app_inf.jl`: functions for approximate inference
    - `dataset.jl`: functions for loading datasets (SYNTH, PHYS101 and ULLMAN)
    - `exp_max.jl`: types and interfaces for the EM algorithm
    - `neural.jl`: generic implementations of neural baselines
    - `sym_reg.jl`: functions for symbolic regression
    - `utility.jl` utility functions for simulation and loss computation
- `Manifest.toml`: the exact package version of this environment
- `master.jl`: master scripts to run a batch of experiments, calling scripts in `scripts/`
- `Project.toml`: the dependency of this environment
- `README.md`: this file

## Setups

BSP is implemented with Julia and some of the dependencies or scripts also rely on Python.

### Julia

Please follow https://julialang.org/downloads/ to download and install Julia.
Make sure `julia` is avaiable in your executable path.
Then from the root of this repo, you can do `julia -e "import Pkg; Pkg.instantiate()"` to instantiate the environment.

There are a few more steps to have Julia properly linked with Python, which is in the next section.

### Python

You will have Python installed and a virtual environment setup.
The virtual environment should have the following packages
- `matplotlib`
- `pandas`
- `wandb`
To properly link this virtual environment with Julia, please follow https://github.com/JuliaPy/PyCall.jl.

You will also have all necessary Python dependencies to run `scripts/preprocess-ullman.py`.
Please see the libraries imported in the script.

## Misc

- Set the environment variable `JULIA_NUM_THREADS=10` before running any scripts will enable multiple-threading (e.g. 10 threads in this example) whenever it's programmed to do so.
    - For example, `master.jl` executes a batch of experiments and is programmed to run them in a multi-threading manner.

