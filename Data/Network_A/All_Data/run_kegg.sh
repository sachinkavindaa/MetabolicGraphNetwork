#!/bin/bash -l
#SBATCH --job-name=kegg_fetch
#SBATCH --partition=guest,batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=10:00:00
#SBATCH --output=kegg.%J.out
#SBATCH --error=kegg.%J.err


cd /work/samodha/sachin/MWAS/All_Data
python -u 01_Kegg.py
