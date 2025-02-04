"""
Author : José CUNHA TEIXEIRA
Affiliation : SNCF Réseau, UMR 7619 METIS (Sorbonne University), Mines Paris - PSL
License : Creative Commons Attribution 4.0 International
Date : Feb 4, 2025
"""

import os


work_dir = f"."
input_dir = f"./input"
output_dir = f"./output"

if not os.path.exists(f"{input_dir}"):
    os.makedirs(f"{input_dir}")
    
if not os.path.exists(f"{output_dir}"):
    os.makedirs(f"{output_dir}")