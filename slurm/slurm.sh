#!/bin/bash
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
module restore
eval "$(conda shell.bash hook)"
conda activate playground
cd ..
# execute some python codes

# gradient matching distillation

# python -u gm_run.py --spc 10
# python -u gm_run.py --spc 100
# python -u gm_run.py --spc 50

# coreset
# python -u gm_run.py --n 1 --obj los
# python -u gm_run.py --n 10 --obj los
# python -u gm_run.py --n 100 --obj los

cd src
python -u main.py