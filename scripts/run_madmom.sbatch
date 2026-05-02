#!/bin/bash
#SBATCH --job-name=madmom_jazz
#SBATCH --output=madmom_%j.out
#SBATCH --error=madmom_%j.err
#SBATCH --account=torch_pr_1003_general
#SBATCH --time=04:00:00          # 4 hours should be plenty
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
# Note: No --gres=gpu requested since Madmom is CPU-only

# Print some info to the log
echo "Starting Madmom Evaluation on Jazz Trio Database..."
echo "Job ID: $SLURM_JOB_ID"
date

# Navigate to your project directory
cd /scratch/yd3245/MIR-Jazz-Beat-Tracking/

# Make sure the results directory exists
mkdir -p results

# Run the Singularity container (without --nv)
singularity exec \
  --bind /scratch:/scratch \
  --overlay /scratch/yd3245/training_project/overlay-25GB.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/miniconda3/bin/activate && python evaluate_madmom.py --data-root /scratch/yd3245/mir_datasets/jtd/annotations --audio-root /scratch/rht9410/jtd_data --output results/madmom_jtd.csv --downbeats"

echo "Evaluation finished!"
date
