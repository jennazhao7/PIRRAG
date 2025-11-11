#!/bin/bash
#$ -N mamba_kd_train                # Job name
#$ -q gpu@@jung_gpu                 # Run on Tjung GPU queue
#$ -pe smp 4                        # 4 CPU cores for data loading
#$ -l gpu_card=1                    # Request 1 GPU (RTX A6000)
#$ -l gpu_mem=48G                   # GPU memory (48 GB on Tjung)
#$ -l h_vmem=32G                    # Host (CPU) memory
#$ -l mem_free=32G
#$ -l h_rt=02:00:00                 # Wall time (2 hours is plenty)
#$ -cwd                             # Run job in current directory
#$ -m abe                           # Email alerts (abort, begin, end)
#$ -M jzhao7@nd.edu                 # Your ND email for notifications
#$ -V                               # Export environment variables

# === Environment setup ===
module load conda
conda activate pdpo
cd /users/jzhao7/MambaDistill       # <-- change to your project folder

# === (Optional) Sync repo if versioned ===
# git fetch origin main
# git reset --hard origin/main

echo "[Step 1] Dumping teacher logits..."
python dump_teacher.py > logs/dump_teacher_$(date +%Y%m%d_%H%M).log 2>&1

echo "[Step 2] Training Tiny Mamba with KD..."
python train_mamba_kd.py > logs/train_mamba_kd_$(date +%Y%m%d_%H%M).log 2>&1

echo "=== Done ==="
