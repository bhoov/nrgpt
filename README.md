# NRGPT

Code supplementary to accompany the NRGPT paper

## Getting started

### **Initialize Environment**

```
uv sync
source .venv/bin/activate
make data # takes a bit
```

All experiments use `wandb` by default. Create a `.env` file in the root directory with your wandb information:

```
WANDB_ENTITY=...
```

## Experiments

### Shakespeare Scaling (SSCL00)

`sweeps/SSCL00_shakespeare_scaling.py`

```
uv run python run_sweep_wandb.py --sweep ./sweeps/SSCL00_tst.py --num_agents 1 --gpus 0
```

### Listops transitions (LOPS00)

`sweeps/LOPS00_listops_training.py`

```
uv run python run_sweep_wandb.py --sweep ./sweeps/LOPS00_listops_training.py --num_agents 1 --gpus 0
```

### Listops deep train (LOPS01)

`sweeps/LOPS01_listops_deep.py`

```
uv run python run_sweep_wandb.py --sweep ./sweeps/LOPS01_listops_deep.py --num_agents 1 --gpus 0
```

### OpenWebText 

```
uv run python run_sweep_wandb.py --sweep ./sweeps/OWT00_owt_training.py --num_agents 1 --allgpus
```
