# SleepAssociationsHPP
This project holds source code used to generate figures and analyses in the manuscript titled: "Phenome Wide Associations of Sleep Monitoring Features in the Human Phenotype project"

# Repo Contents
- [body_systems_associations](./body_systems_associations): Tables with all predictive models results for associations between sleep and other body systems data (pearson R scores), and corresponding figures.
- [clinical_OSA_associations](./clinical_OSA_associations): Tables with logistic regression results for associations between body systems and clinical OSA (estimated from pAHI and daytime sleepiness), and corresponding figures.
- [correlations_results](./correlations_results): Tables with all pairwise correlation results (spearman correlations and p-values), and corresponding figures.
- [descriptive_data_and_figures](./descriptive_data_and_figures): All descriptive figures and tables included in the paper.
- [medical_diagnoses_associations](./medical_diagnoses_associations): Tables with logistic regression results for associations between sleep and medical diagnoses (ROC AUC scores), and corresponding figures.
- [plot_nested_pie.py](./plot_nested_pie.py): Source code for Figure 3 in paper.
- [plot_regression_results.py](./plot_regression_results.py): Source code for Figure 4 and Extended data Figure 4-5 in paper.
- [plot_diagnoses_associations.py](./plot_diagnoses_associations.py): Source code for Figure 5 in paper.
- [plot_clinical_OSA_associations.py](./plot_clinical_OSA_associations.py): Source code for Extended Data Figure 6 in paper.

# System Requirements
## OS Requirements
The version of this source code has been tested on the following systems:   
Linux: CentOS 7(core)  
Mac OS: Monterey 12.5.1   

## Python Dependencies
Make sure your interpreter is set to Python 3.9 and include the packages specified in the [requirements.txt](./requirements.txt) file.    
The code may work properly on other python and packages versions but tested only on these ones.   

# Instructions for use
All associations analyses done as part of this work are available in the respective directories, i.e. Spearman correlations, predictive models scores etc..    
The source code used to evaluate significance (e.g. FDR correction or applying T-test to compare distribution from a simple models) and generate the figures in the paper are available in the root directory.     
The run may take few seconds up to few minutes depending on the number of features and samples in the dataset.   

## Demo
Run file [plot_nested_pie.py](./plot_nested_pie.py) as it is.   
You'll get the following picture as an output:   
![Alt text](./correlations_results/nested_pie-correlations.png)
