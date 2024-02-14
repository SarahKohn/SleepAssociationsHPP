import os

# Define if running in DEMO mode (with sample data / no cluster queue management):
DEMO = True

# Define directory for the data
if DEMO:
    MY_DIR = os.getcwd()
    PATH_FOR_CSV = os.path.join(MY_DIR, 'sample_data')
else:
    from LabData import config_global as config
    from LabUtils.Utils import mkdirifnotexists
    from LabQueue.qp import qp
    from LabUtils.addloglevels import sethandlers

    MY_DIR = os.path.join(config.jafar_base, 'Sarah')
    PATH_FOR_CSV = mkdirifnotexists(os.path.join(MY_DIR, 'csv_files'))

# General globals for plots
REF_COLOR = "k"
FEMALE_COLOR = "C1"
MALE_COLOR = "C0"
ALL_COLOR = "C5"
FONTSIZE = 15
