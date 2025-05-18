#!/bin/bash
#SBATCH --gres=gpu:v100:1                 # Request 1 GPU
#SBATCH --mem=40000
#SBATCH -t 5760
#SBATCH --mail-type=END
#SBATCH -e job-%j.err	

# python collect_predictions_final.py --mask none --cam_mask none
# python collect_predictions_final.py --mask none --cam_mask mask
# python collect_predictions_final.py --mask none --cam_mask lae

# python collect_predictions_final.py --mask mask --cam_mask none
# python collect_predictions_final.py --mask mask --cam_mask mask
# python collect_predictions_final.py --mask mask --cam_mask lae

# python collect_predictions_final.py --mask lae --cam_mask none
# python collect_predictions_final.py --mask lae --cam_mask mask
# python collect_predictions_final.py --mask lae --cam_mask lae

python collect_predictions_standalone.py --mask none 

python collect_predictions_standalone.py --mask mask 

python collect_predictions_standalone.py --mask lae 
