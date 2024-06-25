# Setup
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from globals import *
from matplotlib import colors as mcolors
from matplotlib.colors import Normalize
from mne.stats import fdr_correction


def mkdirifnotexists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def load_dataset_to_name() -> dict:
    return {'Age_Gender_BMI': 'Age & BMI',
            'Age_Gender_BMI_VAT': 'Age, BMI & VAT',
            'hematopoietic': 'Hematopoietic system',
            'immune_system': 'Immune system',
            'glycemic_status': 'Insulin resistance',
            'lifestyle': 'Lifestyle',
            'mental': 'Mental health',
            'frailty': 'Frailty',
            'liver': 'Liver health',
            'renal_function': 'Renal function',
            'cardiovascular': 'Cardiovascular system',
            'body_composition': 'Body Composition',
            'bone_density': 'Bone Density',
            'blood_lipids': 'Blood lipids',
            'medications': 'Medications',
            'MBspecies2': f'Gut MB species',
            'MBgenus2': f'Gut MB genus',
            'MBfamily2': f'Gut MB families',
            'MBphylum2': f'Gut MB phylum',
            'MBclass2': f'Gut MB class',
            'MBorder2': f'Gut MB order',
            'MBpathways': f'Gut MB metabolic pathways',
            'sleep': 'Sleep characteristics',
            'pheno_sleep': f'Sleep characteristics per night\n(with HRV)',
            'pheno_sleep_avg': f'Sleep characteristics\n(average of nights)',
            'sleep_quality_avg': f'Sleep Quality\n(average of nights)',
            'sleep_quality_filtered_avg': f'Sleep Quality\n(average of stable nights)',
            'sleep_quality_best_night': f'Sleep Quality\n(longest night)',
            'hrv_avg': f'HRV\n(average of nights)',
            'diet': 'Diet',
            'all_body_systems': f'All Body Systems\n(except sleep and Genetics)',
            'baseline_diagnoses': 'Diseases and medical conditions\nat baseline',
            'baseline_sleep_quality_avg': f'Sleep Quality at baseline\n',
            'baseline_hrv_avg': f'HRV at baseline\n'
            }


def features_renaming_dict() -> dict:
    return {
        'high_exercise_duration_Between an hour and an hour and a half':
            'high exercise duration',
        'falling_asleep_during_daytime_From time to time': 'falling asleep during daytime',
        'tobacco_past_how_often_I smoked most or all days': 'smoked tobacco most days',
        'Progestogensandestrogenssystemiccontraceptivessequentialpreparations': 'Oral contraceptives',
        'ProtonpumpinhibitorsforpepticulcerandGORD': 'Proton pump inhibitors',
        'Preparationsinhibitinguricacidproduction': 'Uric acid production inhibitors',
        'Dihydropyridinederivativeselectivecalciumchannelblockerswithmainlyvasculareffects':
            'Dihydropyridines calcium channel blockers',
        'AngiotensinIIreceptorblockersARBsplain': 'Angiotensin receptor blockers',
        'Plateletaggregationinhibitorsexclheparin': 'Anti platlets',
        'Betablockingagentsselective': 'Beta blockers',
        'AngiotensinIIreceptorblockersARBsandcalciumchannelblockers':
            'Combination drug - ARBs & calcium channel blockers',
        'ACEinhibitorsplain': 'ACE Inhibitors',
        'HMGCoAreductaseinhibitorsplainlipidmodifyingdrugs': 'Statins',
        'bmi': 'BMI',
        'rds score': 'RDS score',
        'gender': 'sex'
    }


def mbfamily_to_name() -> dict:
    """ Map microbiome families (feature names) into known family names"""
    # from LabData.DataLoaders.GutMBLoader import GutMBLoader
    # gut_mb_loader = GutMBLoader()
    # metadata = gut_mb_loader.get_data('segal_family', study_ids='10K').df_columns_metadata['family'].to_dict()
    return {'fBin__1': 'Methanomassiliicoccaceae',
     'fBin__2': 'Methanomethylophilaceae',
     'fBin__3': 'Methanomethylophilaceae',
     'fBin__4': 'Peptoniphilaceae',
     'fBin__5': 'Helcococcaceae',
     'fBin__6': 'UBA4248',
     'fBin__7': 'CAG-1252',
     'fBin__8': 'Treponemataceae',
     'fBin__9': 'Treponemataceae',
     'fBin__10': 'UBA9783',
     'fBin__11': 'UBA4705',
     'fBin__12': 'UBA4705',
     'fBin__13': 'CAG-312',
     'fBin__14': 'CAG-312',
     'fBin__15': 'Paludibacteraceae',
     'fBin__16': 'UBA932',
     'fBin__17': 'UBA932',
     'fBin__18': 'UBA932',
     'fBin__19': 'UBA932',
     'fBin__20': 'UBA932',
     'fBin__21': 'UBA932',
     'fBin__22': 'UBA932',
     'fBin__23': 'P3',
     'fBin__24': 'F082',
     'fBin__25': 'P3',
     'fBin__26': 'no_consensus',
     'fBin__27': 'Marinifilaceae',
     'fBin__28': 'Marinifilaceae',
     'fBin__29': 'Porphyromonadaceae',
     'fBin__30': 'Porphyromonadaceae',
     'fBin__31': 'Porphyromonadaceae',
     'fBin__32': 'Porphyromonadaceae',
     'fBin__33': 'Muribaculaceae',
     'fBin__34': 'Bacteroidaceae',
     'fBin__35': 'Bacteroidaceae',
     'fBin__36': 'Bacteroidaceae',
     'fBin__37': 'Bacteroidaceae',
     'fBin__38': 'Bacteroidaceae',
     'fBin__39': 'Bacteroidaceae',
     'fBin__40': 'Bacteroidaceae',
     'fBin__41': 'Bacteroidaceae',
     'fBin__42': 'no_consensus',
     'fBin__43': 'Bacteroidaceae',
     'fBin__44': 'Bacteroidaceae',
     'fBin__45': 'Bacteroidaceae',
     'fBin__46': 'Bacteroidaceae',
     'fBin__47': 'Bacteroidaceae',
     'fBin__48': 'Bacteroidaceae',
     'fBin__49': 'Muribaculaceae',
     'fBin__50': 'Bacteroidaceae',
     'fBin__51': 'Muribaculaceae',
     'fBin__52': 'Muribaculaceae',
     'fBin__53': 'Muribaculaceae',
     'fBin__54': 'Muribaculaceae',
     'fBin__55': 'Muribaculaceae',
     'fBin__56': 'Muribaculaceae',
     'fBin__57': 'Muribaculaceae',
     'fBin__58': 'Muribaculaceae',
     'fBin__59': 'Muribaculaceae',
     'fBin__60': 'Muribaculaceae',
     'fBin__61': 'Muribaculaceae',
     'fBin__62': 'Porphyromonadaceae',
     'fBin__63': 'Porphyromonadaceae',
     'fBin__64': 'Muribaculaceae',
     'fBin__65': 'Paludibacteraceae',
     'fBin__66': 'no_consensus',
     'fBin__67': 'Barnesiellaceae',
     'fBin__68': 'Muribaculaceae',
     'fBin__69': 'UBA11471',
     'fBin__70': 'Coprobacteraceae',
     'fBin__71': 'no_consensus',
     'fBin__72': 'Bacteroidaceae',
     'fBin__73': 'Bacteroidaceae',
     'fBin__74': 'Bacteroidaceae',
     'fBin__75': 'Bacteroidaceae',
     'fBin__76': 'Tannerellaceae',
     'fBin__77': 'WCHB1-69',
     'fBin__78': 'UBA1067',
     'fBin__79': 'W1P29-020',
     'fBin__80': 'UBA953',
     'fBin__81': 'UBA953',
     'fBin__82': 'UBA932',
     'fBin__83': 'UBA1820',
     'fBin__84': 'Rikenellaceae',
     'fBin__85': 'Rikenellaceae',
     'fBin__86': 'Rikenellaceae',
     'fBin__87': 'Rikenellaceae',
     'fBin__88': 'Rikenellaceae',
     'fBin__89': 'unknown',
     'fBin__90': 'Elusimicrobiaceae',
     'fBin__91': 'UBA3637',
     'fBin__92': 'unknown',
     'fBin__93': 'unknown',
     'fBin__94': 'Burkholderiaceae',
     'fBin__95': 'Burkholderiaceae',
     'fBin__96': 'Burkholderiaceae',
     'fBin__97': 'Burkholderiaceae',
     'fBin__98': 'Burkholderiaceae',
     'fBin__99': 'Burkholderiaceae',
     'fBin__100': 'Burkholderiaceae',
     'fBin__101': 'Akkermansiaceae',
     'fBin__102': 'Akkermansiaceae',
     'fBin__103': 'Victivallaceae',
     'fBin__104': 'no_consensus',
     'fBin__105': 'UBA1829',
     'fBin__106': 'CAG-977',
     'fBin__107': 'CAG-239',
     'fBin__108': 'CAG-239',
     'fBin__109': 'Vibrionaceae',
     'fBin__110': 'Neisseriaceae',
     'fBin__111': 'SFHR01',
     'fBin__112': 'Actinomycetaceae',
     'fBin__113': 'Neisseriaceae',
     'fBin__114': 'Burkholderiaceae',
     'fBin__115': 'Rhodocyclaceae',
     'fBin__116': 'no_consensus',
     'fBin__117': 'Xanthobacteraceae',
     'fBin__118': 'no_consensus',
     'fBin__119': 'Rs-D84',
     'fBin__120': 'Bifidobacteriaceae',
     'fBin__121': 'Bifidobacteriaceae',
     'fBin__122': 'Bifidobacteriaceae',
     'fBin__123': 'Bifidobacteriaceae',
     'fBin__124': 'Mycobacteriaceae',
     'fBin__125': 'Mycobacteriaceae',
     'fBin__126': 'Mycobacteriaceae',
     'fBin__127': 'Micrococcaceae',
     'fBin__128': 'Propionibacteriaceae',
     'fBin__129': 'no_consensus',
     'fBin__130': 'Actinomycetaceae',
     'fBin__131': 'Actinomycetaceae',
     'fBin__132': 'Actinomycetaceae',
     'fBin__133': 'Actinomycetaceae',
     'fBin__134': 'Actinomycetaceae',
     'fBin__135': 'Flavobacteriaceae',
     'fBin__136': 'Weeksellaceae',
     'fBin__137': 'Methanocorpusculaceae',
     'fBin__138': 'Dysgonomonadaceae',
     'fBin__139': 'Paenibacillaceae',
     'fBin__140': 'Paenibacillaceae',
     'fBin__141': 'Paenibacillaceae',
     'fBin__142': 'DTU023',
     'fBin__143': 'Moraxellaceae',
     'fBin__144': 'UBA932',
     'fBin__145': 'Pasteurellaceae',
     'fBin__146': 'Aeromonadaceae',
     'fBin__147': 'Enterobacteriaceae',
     'fBin__148': 'Enterobacteriaceae',
     'fBin__149': 'Enterobacteriaceae',
     'fBin__150': 'Succinivibrionaceae',
     'fBin__151': 'Succinivibrionaceae',
     'fBin__152': 'Succinivibrionaceae',
     'fBin__153': 'Succinivibrionaceae',
     'fBin__154': 'Lactobacillaceae',
     'fBin__155': 'Lactobacillaceae',
     'fBin__156': 'Lactobacillaceae',
     'fBin__157': 'Lactobacillaceae',
     'fBin__158': 'Lactobacillaceae',
     'fBin__159': 'Lactobacillaceae',
     'fBin__160': 'Lactobacillaceae',
     'fBin__161': 'Lactobacillaceae',
     'fBin__162': 'Lactobacillaceae',
     'fBin__163': 'Lactobacillaceae',
     'fBin__164': 'Lactobacillaceae',
     'fBin__165': 'Lactobacillaceae',
     'fBin__166': 'Brevibacteriaceae',
     'fBin__167': 'Atopobiaceae',
     'fBin__168': 'Lactobacillaceae',
     'fBin__169': 'Lactobacillaceae',
     'fBin__170': 'Peptostreptococcaceae',
     'fBin__171': 'no_consensus',
     'fBin__172': 'Campylobacteraceae',
     'fBin__173': 'Campylobacteraceae',
     'fBin__174': 'Campylobacteraceae',
     'fBin__175': 'Campylobacteraceae',
     'fBin__176': 'Helicobacteraceae',
     'fBin__177': 'Helicobacteraceae',
     'fBin__178': 'Helicobacteraceae',
     'fBin__179': 'Metamycoplasmataceae',
     'fBin__180': 'Bacteroidaceae_A',
     'fBin__181': 'Methanobacteriaceae',
     'fBin__182': 'Methanobacteriaceae',
     'fBin__183': 'Methanobacteriaceae',
     'fBin__184': 'Aerococcaceae',
     'fBin__185': 'Vagococcaceae',
     'fBin__186': 'Enterococcaceae',
     'fBin__187': 'Streptococcaceae',
     'fBin__188': 'Streptococcaceae',
     'fBin__189': 'Turicibacteraceae',
     'fBin__190': 'Planococcaceae',
     'fBin__191': 'Exiguobacteraceae',
     'fBin__192': 'Bacillaceae_G',
     'fBin__193': 'no_consensus',
     'fBin__194': 'Bacillaceae_A',
     'fBin__195': 'Gemellaceae',
     'fBin__196': 'no_consensus',
     'fBin__197': 'Bacillaceae_A',
     'fBin__198': 'CAG-826',
     'fBin__199': 'CAG-826',
     'fBin__200': 'CAG-826',
     'fBin__201': 'CAG-826',
     'fBin__202': 'CAG-826',
     'fBin__203': 'CAG-826',
     'fBin__204': 'CAG-288',
     'fBin__205': 'Erysipelotrichaceae',
     'fBin__206': 'Erysipelotrichaceae',
     'fBin__207': 'CAG-288',
     'fBin__208': 'CAG-288',
     'fBin__209': 'CAG-288',
     'fBin__210': 'CAG-288',
     'fBin__211': 'CAG-433',
     'fBin__212': 'CAG-302',
     'fBin__213': 'CAG-449',
     'fBin__214': 'Anaeroplasmataceae',
     'fBin__215': 'Anaeroplasmataceae',
     'fBin__216': 'Erysipelatoclostridiaceae',
     'fBin__217': 'Erysipelatoclostridiaceae',
     'fBin__218': 'Erysipelatoclostridiaceae',
     'fBin__219': 'Erysipelotrichaceae',
     'fBin__220': 'Erysipelotrichaceae',
     'fBin__221': 'Erysipelotrichaceae',
     'fBin__222': 'Erysipelotrichaceae',
     'fBin__223': 'Erysipelotrichaceae',
     'fBin__224': 'Erysipelotrichaceae',
     'fBin__225': 'Erysipelotrichaceae',
     'fBin__226': 'Erysipelotrichaceae',
     'fBin__227': 'Erysipelotrichaceae',
     'fBin__228': 'Ruminococcaceae',
     'fBin__229': 'UBA3375',
     'fBin__230': 'Peptoniphilaceae',
     'fBin__231': 'Filifactoraceae',
     'fBin__232': 'Ezakiellaceae',
     'fBin__233': 'Filifactoraceae',
     'fBin__234': 'Ezakiellaceae',
     'fBin__235': 'Peptoniphilaceae',
     'fBin__236': 'Helcococcaceae',
     'fBin__237': 'no_consensus',
     'fBin__238': 'Peptoniphilaceae',
     'fBin__239': 'Helcococcaceae',
     'fBin__240': 'Garciellaceae',
     'fBin__241': 'Sporanaerobacteraceae',
     'fBin__242': 'Brachyspiraceae',
     'fBin__243': 'Leptotrichiaceae',
     'fBin__244': 'Leptotrichiaceae',
     'fBin__245': 'Fusobacteriaceae',
     'fBin__246': 'Fusobacteriaceae',
     'fBin__247': 'Peptostreptococcaceae',
     'fBin__248': 'Clostridiaceae',
     'fBin__249': 'Clostridiaceae',
     'fBin__250': 'Clostridiaceae',
     'fBin__251': 'Cellulosilyticaceae',
     'fBin__252': 'Lachnospiraceae',
     'fBin__253': 'CAG-611',
     'fBin__254': 'CAG-611',
     'fBin__255': 'CAG-611',
     'fBin__256': 'UBA1234',
     'fBin__257': 'no_consensus',
     'fBin__258': 'CAG-465',
     'fBin__259': 'unknown',
     'fBin__260': 'CAG-822',
     'fBin__261': 'CAG-822',
     'fBin__262': 'no_consensus',
     'fBin__263': 'no_consensus',
     'fBin__264': 'CAG-611',
     'fBin__265': 'no_consensus',
     'fBin__266': 'Mycoplasmoidaceae',
     'fBin__267': 'Mycoplasmoidaceae',
     'fBin__268': 'CAG-631',
     'fBin__269': 'CAG-631',
     'fBin__270': 'CAG-313',
     'fBin__271': 'CAG-698',
     'fBin__272': 'Erysipelotrichaceae',
     'fBin__273': 'Anaeroplasmataceae',
     'fBin__274': 'no_consensus',
     'fBin__275': 'no_consensus',
     'fBin__276': 'CAG-826',
     'fBin__277': 'CAG-449',
     'fBin__278': 'CAG-288',
     'fBin__279': 'CAG-826',
     'fBin__280': 'Gastranaerophilaceae',
     'fBin__281': 'Gastranaerophilaceae',
     'fBin__282': 'Gastranaerophilaceae',
     'fBin__283': 'no_consensus',
     'fBin__284': 'Gastranaerophilaceae',
     'fBin__285': 'unknown',
     'fBin__286': 'Gastranaerophilaceae',
     'fBin__287': 'Acutalibacteraceae',
     'fBin__288': 'Fastidiosipilaceae',
     'fBin__289': 'Acutalibacteraceae',
     'fBin__290': 'Sporolactobacillaceae',
     'fBin__291': 'Sedimentibacteraceae',
     'fBin__292': 'UBA4877',
     'fBin__293': 'no_consensus',
     'fBin__294': 'Helcococcaceae',
     'fBin__295': 'Helcococcaceae',
     'fBin__296': 'QAND01',
     'fBin__297': 'Lachnospiraceae',
     'fBin__298': 'Acutalibacteraceae',
     'fBin__299': 'Actinomycetaceae',
     'fBin__300': 'Atopobiaceae',
     'fBin__301': 'Atopobiaceae',
     'fBin__302': 'Atopobiaceae',
     'fBin__303': 'Atopobiaceae',
     'fBin__304': 'Eggerthellaceae',
     'fBin__305': 'Eggerthellaceae',
     'fBin__306': 'Eggerthellaceae',
     'fBin__307': 'Eggerthellaceae',
     'fBin__308': 'Acidaminococcaceae',
     'fBin__309': 'Acidaminococcaceae',
     'fBin__310': 'Acidaminococcaceae',
     'fBin__311': 'Acidaminococcaceae',
     'fBin__312': 'no_consensus',
     'fBin__313': 'Veillonellaceae',
     'fBin__314': 'Veillonellaceae',
     'fBin__315': 'Selenomonadaceae',
     'fBin__316': 'Negativicoccaceae',
     'fBin__317': 'Selenomonadaceae',
     'fBin__318': 'Selenomonadaceae',
     'fBin__319': 'Dialisteraceae',
     'fBin__320': 'Dialisteraceae',
     'fBin__321': 'Megasphaeraceae',
     'fBin__322': 'Megasphaeraceae',
     'fBin__323': 'Anaerovoracaceae',
     'fBin__324': 'Anaerovoracaceae',
     'fBin__325': 'Anaerovoracaceae',
     'fBin__326': 'Anaerovoracaceae',
     'fBin__327': 'Anaerovoracaceae',
     'fBin__328': 'Anaerovoracaceae',
     'fBin__329': 'Anaerovoracaceae',
     'fBin__330': 'Anaerovoracaceae',
     'fBin__331': 'Anaerovoracaceae',
     'fBin__332': 'Anaerovoracaceae',
     'fBin__333': 'Anaerovoracaceae',
     'fBin__334': 'Anaerovoracaceae',
     'fBin__335': 'CAG-272',
     'fBin__336': 'Anaerovoracaceae',
     'fBin__337': 'Anaerovoracaceae',
     'fBin__338': 'Anaerovoracaceae',
     'fBin__339': 'Anaerotignaceae',
     'fBin__340': 'Anaerotignaceae',
     'fBin__341': 'Anaerotignaceae',
     'fBin__342': 'no_consensus',
     'fBin__343': 'Lachnospiraceae',
     'fBin__344': 'Lachnospiraceae',
     'fBin__345': 'Lachnospiraceae',
     'fBin__346': 'Lachnospiraceae',
     'fBin__347': 'Lachnospiraceae',
     'fBin__348': 'Lachnospiraceae',
     'fBin__349': 'no_consensus',
     'fBin__350': 'Lachnospiraceae',
     'fBin__351': 'Peptococcaceae',
     'fBin__352': 'Lachnospiraceae',
     'fBin__353': 'Lachnospiraceae',
     'fBin__354': 'Lachnospiraceae',
     'fBin__355': 'Lachnospiraceae',
     'fBin__356': 'Lachnospiraceae',
     'fBin__357': 'Lachnospiraceae',
     'fBin__358': 'Lachnospiraceae',
     'fBin__359': 'Lachnospiraceae',
     'fBin__360': 'Lachnospiraceae',
     'fBin__361': 'Lachnospiraceae',
     'fBin__362': 'Lachnospiraceae',
     'fBin__363': 'Lachnospiraceae',
     'fBin__364': 'Lachnospiraceae',
     'fBin__365': 'Lachnospiraceae',
     'fBin__366': 'Lachnospiraceae',
     'fBin__367': 'Lachnospiraceae',
     'fBin__368': 'Lachnospiraceae',
     'fBin__369': 'Lachnospiraceae',
     'fBin__370': 'Lachnospiraceae',
     'fBin__371': 'Lachnospiraceae',
     'fBin__372': 'Lachnospiraceae',
     'fBin__373': 'Lachnospiraceae',
     'fBin__374': 'Lachnospiraceae',
     'fBin__375': 'Lachnospiraceae',
     'fBin__376': 'no_consensus',
     'fBin__377': 'Lachnospiraceae',
     'fBin__378': 'Lachnospiraceae',
     'fBin__379': 'Lachnospiraceae',
     'fBin__380': 'Lachnospiraceae',
     'fBin__381': 'Lachnospiraceae',
     'fBin__382': 'Lachnospiraceae',
     'fBin__383': 'Lachnospiraceae',
     'fBin__384': 'Lachnospiraceae',
     'fBin__385': 'Lachnospiraceae',
     'fBin__386': 'Lachnospiraceae',
     'fBin__387': 'Lachnospiraceae',
     'fBin__388': 'Lachnospiraceae',
     'fBin__389': 'Lachnospiraceae',
     'fBin__390': 'Lachnospiraceae',
     'fBin__391': 'Lachnospiraceae',
     'fBin__392': 'Lachnospiraceae',
     'fBin__393': 'CAG-274',
     'fBin__394': 'CAG-274',
     'fBin__395': 'UBA1390',
     'fBin__396': 'UBA1390',
     'fBin__397': 'Victivallaceae',
     'fBin__398': 'UBA1242',
     'fBin__399': 'UBA3700',
     'fBin__400': 'UBA1242',
     'fBin__401': 'UBA1242',
     'fBin__402': 'UBA1242',
     'fBin__403': 'UBA1242',
     'fBin__404': 'UMGS1908',
     'fBin__405': 'UBA1242',
     'fBin__406': 'UBA1242',
     'fBin__407': 'UBA1242',
     'fBin__408': 'UBA1242',
     'fBin__409': 'UMGS1908',
     'fBin__410': 'UMGS1908',
     'fBin__411': 'UBA1242',
     'fBin__412': 'CAG-314',
     'fBin__413': 'CAG-314',
     'fBin__414': 'Borkfalkiaceae',
     'fBin__415': 'CAG-314',
     'fBin__416': 'CAG-314',
     'fBin__417': 'UBA3700',
     'fBin__418': 'CAG-314',
     'fBin__419': 'CAG-314',
     'fBin__420': 'Christensenellaceae',
     'fBin__421': 'Christensenellaceae',
     'fBin__422': 'Christensenellaceae',
     'fBin__423': 'QALW01',
     'fBin__424': 'QALW01',
     'fBin__425': 'no_consensus',
     'fBin__426': 'Christensenellaceae',
     'fBin__427': 'GCA-900066905',
     'fBin__428': 'unknown',
     'fBin__429': 'no_consensus',
     'fBin__430': 'UBA1381',
     'fBin__431': 'UBA9506',
     'fBin__432': 'no_consensus',
     'fBin__433': 'Monoglobaceae',
     'fBin__434': 'UBA9506',
     'fBin__435': 'unknown',
     'fBin__436': 'UMGS1253',
     'fBin__437': 'unknown',
     'fBin__438': 'Monoglobaceae',
     'fBin__439': 'unknown',
     'fBin__440': 'UBA9506',
     'fBin__441': 'no_consensus',
     'fBin__442': 'unknown',
     'fBin__443': 'UMGS1810',
     'fBin__444': 'UBA1381',
     'fBin__445': 'Borkfalkiaceae',
     'fBin__446': 'Borkfalkiaceae',
     'fBin__447': 'DTU072',
     'fBin__448': 'no_consensus',
     'fBin__449': 'CAG-552',
     'fBin__450': 'Borkfalkiaceae',
     'fBin__451': 'Borkfalkiaceae',
     'fBin__452': 'Borkfalkiaceae',
     'fBin__453': 'Borkfalkiaceae',
     'fBin__454': 'Borkfalkiaceae',
     'fBin__455': 'Borkfalkiaceae',
     'fBin__456': 'CAG-552',
     'fBin__457': 'CAG-552',
     'fBin__458': 'CAG-552',
     'fBin__459': 'CAG-917',
     'fBin__460': 'CAG-917',
     'fBin__461': 'CAG-917',
     'fBin__462': 'CAG-917',
     'fBin__463': 'CAG-917',
     'fBin__464': 'UBA644',
     'fBin__465': 'QAKW01',
     'fBin__466': 'UBA644',
     'fBin__467': 'CAG-272',
     'fBin__468': 'CAG-382',
     'fBin__469': 'CAG-272',
     'fBin__470': 'CAG-272',
     'fBin__471': 'CAG-382',
     'fBin__472': 'CAG-272',
     'fBin__473': 'CAG-272',
     'fBin__474': 'CAG-272',
     'fBin__475': 'CAG-272',
     'fBin__476': 'CAG-272',
     'fBin__477': 'CAG-272',
     'fBin__478': 'CAG-272',
     'fBin__479': 'CAG-272',
     'fBin__480': 'CAG-382',
     'fBin__481': 'UBA644',
     'fBin__482': 'no_consensus',
     'fBin__483': 'CAG-272',
     'fBin__484': 'CAG-272',
     'fBin__485': 'no_consensus',
     'fBin__486': 'CAG-272',
     'fBin__487': 'CAG-272',
     'fBin__488': 'CAG-272',
     'fBin__489': 'CAG-272',
     'fBin__490': 'CAG-272',
     'fBin__491': 'CAG-272',
     'fBin__492': 'Peptococcaceae',
     'fBin__493': 'Peptococcaceae',
     'fBin__494': 'Lachnospiraceae',
     'fBin__495': 'CAG-272',
     'fBin__496': 'Lachnospiraceae',
     'fBin__497': 'Synergistaceae',
     'fBin__498': 'Dethiosulfovibrionaceae',
     'fBin__499': 'Dethiosulfovibrionaceae',
     'fBin__500': 'no_consensus',
     'fBin__501': 'CAG-138',
     'fBin__502': 'CAG-138',
     'fBin__503': 'CAG-138',
     'fBin__504': 'CAG-138',
     'fBin__505': 'CAG-138',
     'fBin__506': 'CAG-138',
     'fBin__507': 'CAG-138',
     'fBin__508': 'Desulfovibrionaceae',
     'fBin__509': 'Desulfovibrionaceae',
     'fBin__510': 'Sphaerochaetaceae',
     'fBin__511': 'Sphaerochaetaceae',
     'fBin__512': 'UBA1407',
     'fBin__513': 'CAG-272',
     'fBin__514': 'Ruminococcaceae',
     'fBin__515': 'Peptococcaceae',
     'fBin__516': 'Atopobiaceae',
     'fBin__517': 'no_consensus',
     'fBin__518': 'QAMH01',
     'fBin__519': 'Eggerthellaceae',
     'fBin__520': 'Oscillospiraceae',
     'fBin__521': 'Oscillospiraceae',
     'fBin__522': 'QAMX01',
     'fBin__523': 'Ruminococcaceae',
     'fBin__524': 'Acutalibacteraceae',
     'fBin__525': 'Acutalibacteraceae',
     'fBin__526': 'Acutalibacteraceae',
     'fBin__527': 'CAG-74',
     'fBin__528': 'UBA1750',
     'fBin__529': 'CAG-74',
     'fBin__530': 'CAG-74',
     'fBin__531': 'CAG-74',
     'fBin__532': 'Acutalibacteraceae',
     'fBin__533': 'Butyricicoccaceae',
     'fBin__534': 'Ruminococcaceae',
     'fBin__535': 'Oscillospiraceae',
     'fBin__536': 'no_consensus',
     'fBin__537': 'no_consensus',
     'fBin__538': 'Ruminococcaceae',
     'fBin__539': 'Oscillospiraceae',
     'fBin__540': 'Oscillospiraceae',
     'fBin__541': 'no_consensus',
     'fBin__542': 'Oscillospiraceae',
     'fBin__543': 'Oscillospiraceae',
     'fBin__544': 'unknown',
     'fBin__545': 'unknown',
     'fBin__546': 'no_consensus',
     'fBin__547': 'UBA1255',
     'fBin__548': 'no_consensus',
     'fBin__549': 'Ruminococcaceae',
     'fBin__550': 'no_consensus',
     'fBin__551': 'Acutalibacteraceae',
     'fBin__552': 'Ruminococcaceae',
     'fBin__553': 'CAG-272',
     'fBin__554': 'Ruminococcaceae',
     'fBin__555': 'CAG-272',
     'fBin__556': 'QALW01',
     'fBin__557': 'QALW01',
     'fBin__558': 'Acutalibacteraceae',
     'fBin__559': 'Lachnospiraceae',
     'fBin__560': 'Ruminococcaceae',
     'fBin__561': 'no_consensus',
     'fBin__562': 'Ruminococcaceae',
     'fBin__563': 'QAKW01',
     'fBin__564': 'Ruminococcaceae',
     'fBin__565': 'Ruminococcaceae',
     'fBin__566': 'UMGS1783',
     'fBin__567': 'Acutalibacteraceae',
     'fBin__568': 'Acutalibacteraceae',
     'fBin__569': 'Ruminococcaceae',
     'fBin__570': 'Acutalibacteraceae',
     'fBin__571': 'no_consensus',
     'fBin__572': 'Acutalibacteraceae',
     'fBin__573': 'Acutalibacteraceae',
     'fBin__574': 'Acutalibacteraceae',
     'fBin__575': 'Acutalibacteraceae',
     'fBin__576': 'Acutalibacteraceae',
     'fBin__577': 'Ruminococcaceae',
     'fBin__578': 'no_consensus',
     'fBin__579': 'Ruminococcaceae',
     'fBin__580': 'Ruminococcaceae',
     'fBin__581': 'Acutalibacteraceae',
     'fBin__582': 'Acutalibacteraceae',
     'fBin__583': 'Ruminococcaceae',
     'fBin__584': 'no_consensus',
     'fBin__585': 'Ruminococcaceae',
     'fBin__586': 'unknown',
     'fBin__587': 'Acutalibacteraceae',
     'fBin__588': 'no_consensus',
     'fBin__589': 'Butyricicoccaceae',
     'fBin__590': 'Butyricicoccaceae',
     'fBin__591': 'Oscillospiraceae',
     'fBin__592': 'Ruminococcaceae',
     'fBin__593': 'Acutalibacteraceae',
     'fBin__594': 'UBA1381',
     'fBin__595': 'Acutalibacteraceae',
     'fBin__596': 'Acutalibacteraceae',
     'fBin__597': 'Ruminococcaceae',
     'fBin__598': 'Ruminococcaceae',
     'fBin__599': 'Ruminococcaceae',
     'fBin__600': 'Ruminococcaceae',
     'fBin__601': 'Ruminococcaceae',
     'fBin__602': 'Ruminococcaceae',
     'fBin__603': 'Ruminococcaceae',
     'fBin__604': 'Ruminococcaceae',
     'fBin__605': 'Acutalibacteraceae',
     'fBin__606': 'Acutalibacteraceae',
     'fBin__607': 'Acutalibacteraceae',
     'fBin__608': 'Acutalibacteraceae',
     'fBin__609': 'Acutalibacteraceae',
     'fBin__610': 'Acutalibacteraceae',
     'fBin__611': 'Acutalibacteraceae',
     'fBin__612': 'Acutalibacteraceae',
     'fBin__613': 'Acutalibacteraceae',
     'fBin__614': 'Acutalibacteraceae',
     'fBin__615': 'Ruminococcaceae',
     'fBin__616': 'Acutalibacteraceae',
     'fBin__617': 'UMGS1883',
     'fBin__618': 'Acutalibacteraceae',
     'fBin__619': 'Acutalibacteraceae',
     'fBin__620': 'Acutalibacteraceae',
     'fBin__621': 'no_consensus',
     'fBin__622': 'Eubacteriaceae',
     'fBin__623': 'Anaerovoracaceae',
     'fBin__624': 'Anaerovoracaceae',
     'fBin__625': 'Acutalibacteraceae',
     'fBin__626': 'Acutalibacteraceae',
     'fBin__627': 'Acutalibacteraceae'}



def remove_cols_with_missing_values(df: pd.DataFrame,
                                    no_of_valid_values_required: float = 500) -> pd.DataFrame:
    """Returns a dataframe identical to the input one, but without the columns that have less than
    the 'no_of_valid_values_required' elements.
    Invalid elements in that case are all the NaN or Inf elements """
    version = sys.version_info
    if version.minor <= 7:
        return df[[col for col in df.columns if ((~df[col].isna()).sum() >= no_of_valid_values_required).all()]]
    else:
        return df[[col for col in df.columns if ((~df[col].isna()).sum() >= no_of_valid_values_required)]]


def remove_cols_with_low_variability(df: pd.DataFrame,
                                     no_of_different_values_required: float = 500) -> pd.DataFrame:
    """ Returns a dataframe identical to the input one, but without the columns that have less than
    the 'no_of_different_values_required' elements with values different from the most prevalant one """
    return df[[col for col in df.columns if (
            (len(df[col].value_counts()) != 1) and
            (df[col].value_counts().max() <= len(df[col]) - no_of_different_values_required))]]


def get_significant_df_after_fdr_correction(df: pd.DataFrame,
                                            pvalues: pd.DataFrame,
                                            alpha: float,
                                            cluster: bool = False) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove non-significant rows/columns of a Dataframe after FDR correction
    returns the updated dataframe, with corresponding pvalues and mask"""
    mask = pd.DataFrame(~fdr_correction(pvalues.stack(), alpha=alpha)[0],
                        index=pvalues.stack().index).unstack()
    mask.columns = [m[-1] for m in mask.columns]
    mask = mask.reindex(pvalues.columns, axis=1)
    mask.fillna(True, inplace=True)
    df[mask] = np.nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.dropna(how='all', axis=0, inplace=True)
    df.fillna(0, inplace=True)
    if cluster:
        # correlate the correlation matrix
        df = reorder_by_clusters(df)
    pvalues = pvalues.loc[df.index, df.columns]
    mask = mask.loc[df.index, df.columns]
    return [df, pvalues, mask]


def reorder_by_clusters(df: pd.DataFrame) -> pd.DataFrame:
    cg1 = sns.clustermap(df, xticklabels=False, yticklabels=False, metric='euclidean')
    plt.clf()
    return df.iloc[cg1.dendrogram_row.reordered_ind, cg1.dendrogram_col.reordered_ind]


def reindex_by_reg_code_and_research_stage(loader_data):
    """Reindex the data and metadata of a dataloader,
    replacing the date by the research stage in the multi-index"""
    loader_data.df = loader_data.df.join(loader_data.df_metadata['research_stage'])
    loader_data.df = loader_data.df.set_index([loader_data.df.index.get_level_values(0), 'research_stage'])
    loader_data.df_metadata = loader_data.df_metadata.set_index(
        [loader_data.df_metadata.index.get_level_values(0), 'research_stage'])
    return loader_data


def get_scale_colors(cmaps, data, zero_is_middle=True, base_n=300, boundries=None, return_cmap=False):
    if boundries is None:
        data_plus_min = data - min(0, data.min())
        data_plus_min /= data_plus_min.max()
        min_max_ratio = abs(data.min() / float(data.max()))
    else:
        data_plus_min = data + abs(boundries[0])
        data_plus_min /= (abs(boundries[0]) + boundries[1])
        min_max_ratio = abs(boundries[0] / float(boundries[1]))
    if len(cmaps) == 1:
        return [cmaps[0](i) for i in data_plus_min]
    colors1 = cmaps[0](np.linspace(0., 1, int(base_n * min_max_ratio)))
    colors2 = cmaps[1](np.linspace(0., 1, base_n))
    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    if return_cmap:
        return mymap
    return [mymap(i) for i in data_plus_min]


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Ignoring masked values and all kinds of edge cases to make a simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
