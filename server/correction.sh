#!/bin/bash
#
#SBATCH -c 8
#SBATCH --time=0-20:00:00
#SBATCH --mem-per-cpu=8G

python neuriteX_correction.py
