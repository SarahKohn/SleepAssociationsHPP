import os
import numpy as np
import seaborn as sns

# Define directory for the data
MY_DIR = os.getcwd()

# Fix seed for random processes
SEED = 18
np.random.seed(18)

# General globals for plots
REF_COLOR = "k"
FEMALE_COLOR = "C1"
MALE_COLOR = "C0"
ALL_COLOR = "C5"
FONTSIZE = 15
DPI = 300