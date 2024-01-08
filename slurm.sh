#!/bin/bash
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
module restore
eval "$(conda shell.bash hook)"
conda activate playground

# execute some python codes

# gradient matching distillation

# python -u gm_run.py --spc 10
# python -u gm_run.py --spc 100
# python -u gm_run.py --spc 50

# coreset
python -u coreset_run.py --spc 1 --nexp 10
python -u coreset_run.py --spc 10 --nexp 10
python -u coreset_run.py --spc 100 --nexp 10