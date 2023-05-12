"""This was a WIP file with data structures for input data. Use model1.py for predictions"""
import os
import re
import hashlib
import random

from sys import version_info 
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Bio import BiopythonDeprecationWarning
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)
from pathlib import Path
from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import setup_logging
from colabfold.batch import get_queries, run, set_model_type
from colabfold.plot import plot_msa_v2

import os
import numpy as np
from colabfold.colabfold import plot_protein
from pathlib import Path
import matplotlib.pyplot as plt
import py3Dmol
import glob
import matplotlib.pyplot as plt
from colabfold.colabfold import plot_plddt_legend
from colabfold.colabfold import pymol_color_list, alphabet_list
import fileinput
from dataclass import dataclass
from enum import Enum, IntEnum
python_version = f"{version_info.major}.{version_info.minor}"


def add_hash(x,y):
    return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]


class AFTemplateType(Enum):
    none = "none"
    pdb70 = "pdb70"
    custom = "custom"

class AFModelType(Enum):
    auto =                     "auto"
    alphafold2_ptm =           "alphafold2_ptm"
    alphafold2_multimer_v1 =   "alphafold2_multimer_v1"
    alphafold2_multimer_v2 =   "alphafold2_multimer_v2"
    alphafold2_multimer_v3 =   "alphafold2_multimer_v3"

class AFNumRecycles(Enum):
    auto =   "auto"
    _0 =     "0"
    _1 =     "1"
    _3 =     "3"
    _6 =     "6"
    _12 =    "12"
    _24 =    "24"
    _48 =    "48"

class AFRecycleEarlyStopTolerance(Enum):
    _auto =   "auto"
    _0_0 =    "0.0"
    _0_5 =    "0.5"
    _1_0 =    "1.0"


class AFMaxMSA(Enum):
    auto =       "auto"
    _512_1024 =   "512:1024"
    _256_512 =    "256:512"
    _64_128 =     "64:128"
    _32_64 =      "32:64"
    _16_32 =      "16:32"

class AFNumSeeds(IntEnu):
    _1 = 1
    _2 = 2
    _4 = 4
    _8 = 8
    _16 = 16




@dataclass
class AlphaFoldData:
    query_string: str
    basejobname: str = "test"
    """# number of models to use"""
    num_relax: int = 0
    """#@markdown - specify how many of the top ranked structures to relax using amber"""
    template_mode: AFTemplateType = AFTemplateType.none
    """#@markdown - `none` = no template information is used. `pdb70` = detect templates in pdb70. `custom` - upload
    and search own templates (PDB or mmCIF format, see [notes below](#custom_templates))"""

    
    # From Advanced settings
    model_type: AFModelType = AFModelType.auto
    #@markdown - if `auto` selected, will use `alphafold2_ptm` for monomer prediction and `alphafold2_multimer_v3` for complex prediction. 
    #@markdown Any of the mode_types can be used (regardless if input is monomer or complex).
    _num_recycles = AFNumRecycles.auto
    recycle_early_stop_tolerance = AFRecycleEarlyStopTolerance.auto
    #@markdown - if `auto` selected, will use 20 recycles if `model_type=alphafold2_multimer_v3` (with tol=0.5), all else 3 recycles (with tol=0.0).

    #@markdown #### Sample settings
    #@markdown -  enable dropouts and increase number of seeds to sample predictions from uncertainty of the model.
    #@markdown -  decrease `max_msa` to increase uncertainity
    max_msa = AFMaxMSA.auto
    num_seeds = AFNumSeeds._1 #@param [1,2,4,8,16] {type:"raw"}
    use_dropout = False #@param {type:"boolean"}

    recycle_early_stop_tolerance = None if recycle_early_stop_tolerance == "auto" else float(recycle_early_stop_tolerance)
    if max_msa == "auto": max_msa = None

    #@markdown #### Save settings
    save_all = False #@param {type:"boolean"}
    save_recycles = False #@param {type:"boolean"}
    save_to_google_drive = False #@param {type:"boolean"}
    #@markdown -  if the save_to_google_drive option was selected, the result zip will be uploaded to your Google Drive
    dpi = 200 #@param {type:"integer"}
    #@markdown - set dpi for image resolution

    @property
    def use_amber(self) -> bool:
        """The use_amber property."""
        return self.num_relax > 0

    @property
    def num_recycles(self):
        return None if self._num_recycles.value == "auto" else int(self._num_recycles.value)


    


def generate_job_name():
    #@markdown  - Use `:` to specify inter-protein chainbreaks for **modeling complexes** (supports homo- and hetro-oligomers). For example **PI...SK:PI...SK** for a homodimer
    jobname = 'test' #@param {type:"string"}
    # number of models to use
    num_relax = 0 #@param [0, 1, 5] {type:"raw"}
    #@markdown - specify how many of the top ranked structures to relax using amber
    template_mode = "none" #@param ["none", "pdb70","custom"]
    #@markdown - `none` = no template information is used. `pdb70` = detect templates in pdb70. `custom` - upload and search own templates (PDB or mmCIF format, see [notes below](#custom_templates))

    use_amber = num_relax > 0

    # remove whitespaces
    query_sequence = "".join(query_sequence.split())

    basejobname = "".join(jobname.split())
    basejobname = re.sub(r'\W+', '', basejobname)
    jobname = add_hash(basejobname, query_sequence)

    # check if directory with jobname exists
    def check(folder):
        if os.path.exists(folder):
            return False
        else:
            return True
        if not check(jobname):
            n = 0
        while not check(f"{jobname}_{n}"):
            n += 1
        jobname = f"{jobname}_{n}"

    # make directory to save results
    os.makedirs(jobname, exist_ok=True)

    # save queries
    queries_path = os.path.join(jobname, f"{jobname}.csv")
    with open(queries_path, "w") as text_file:
        text_file.write(f"id,sequence\n{jobname},{query_sequence}")

    if template_mode == "pdb70":
        use_templates = True
        custom_template_path = None
    elif template_mode == "custom":
        custom_template_path = os.path.join(jobname,f"template")
        os.makedirs(custom_template_path, exist_ok=True)
        uploaded = files.upload()
        use_templates = True
        for fn in uploaded.keys():
            os.rename(fn,os.path.join(custom_template_path,fn))
    else:
        custom_template_path = None
        use_templates = False

    print("jobname",jobname)
    print("sequence",query_sequence)
    print("length",len(query_sequence.replace(":","")))


def pick_a3m_module():
    #@markdown ### MSA options (custom MSA upload, single sequence, pairing mode)
    msa_mode = "mmseqs2_uniref_env" #@param ["mmseqs2_uniref_env", "mmseqs2_uniref","single_sequence","custom"]
    pair_mode = "unpaired_paired" #@param ["unpaired_paired","paired","unpaired"] {type:"string"}
    #@markdown - "unpaired_paired" = pair sequences from same species + unpaired MSA, "unpaired" = seperate MSA for each chain, "paired" - only use paired sequences.

    # decide which a3m to use
    if "mmseqs2" in msa_mode:
        a3m_file = os.path.join(jobname,f"{jobname}.a3m")

    elif msa_mode == "custom":
        a3m_file = os.path.join(jobname,f"{jobname}.custom.a3m")
        if not os.path.isfile(a3m_file):
            custom_msa_dict = files.upload()
            custom_msa = list(custom_msa_dict.keys())[0]
            header = 0
            for line in fileinput.FileInput(custom_msa,inplace=1):
                if line.startswith(">"):
                    header = header + 1
                if not line.rstrip():
                    continue
                if line.startswith(">") == False and header == 1:
                    query_sequence = line.rstrip()
                print(line, end='')

            os.rename(custom_msa, a3m_file)
            queries_path=a3m_file
            print(f"moving {custom_msa} to {a3m_file}")

    else:
        a3m_file = os.path.join(jobname,f"{jobname}.single_sequence.a3m")
        with open(a3m_file, "w") as text_file:
            text_file.write(">1\n%s" % query_sequence)


def predict(query_sequence):
    #@title Run Prediction
    display_images = True #@param {type:"boolean"}

    try:
        K80_chk = os.popen('nvidia-smi | grep "Tesla K80" | wc -l').read()
    except:
        K80_chk = "0"
        pass
    if "1" in K80_chk:
        print("WARNING: found GPU Tesla K80: limited to total length < 1000")
        if "TF_FORCE_UNIFIED_MEMORY" in os.environ:
            del os.environ["TF_FORCE_UNIFIED_MEMORY"]
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
            del os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]


    # For some reason we need that to get pdbfixer to import
    if use_amber and f"/usr/local/lib/python{python_version}/site-packages/" not in sys.path:
        sys.path.insert(0, f"/usr/local/lib/python{python_version}/site-packages/")

    def input_features_callback(input_features):  
        if display_images:    
            plot_msa_v2(input_features)
            plt.show()
            plt.close()

    def prediction_callback(protein_obj, length,
                            prediction_result, input_features, mode):
        model_name, relaxed = mode
        if not relaxed:
            if display_images:
                fig = plot_protein(protein_obj, Ls=length, dpi=150)
                plt.show()
                plt.close()

    result_dir = jobname
    if 'logging_setup' not in globals():
        setup_logging(Path(os.path.join(jobname,"log.txt")))
        logging_setup = True

    queries, is_complex = get_queries(queries_path)
    model_type = set_model_type(is_complex, model_type)

    if "multimer" in model_type and max_msa is not None:
        use_cluster_profile = False
    else:
        use_cluster_profile = True

    download_alphafold_params(model_type, Path("."))
    results = run(
        queries=queries,
        result_dir=result_dir,
        use_templates=use_templates,
        custom_template_path=custom_template_path,
        num_relax=num_relax,
        msa_mode=msa_mode,    
        model_type=model_type,
        num_models=5,
        num_recycles=num_recycles,
        recycle_early_stop_tolerance=recycle_early_stop_tolerance,
        num_seeds=num_seeds,
        use_dropout=use_dropout,
        model_order=[1,2,3,4,5],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        rank_by="auto",
        pair_mode=pair_mode,
        stop_at_score=float(100),
        prediction_callback=prediction_callback,
        dpi=dpi,
        zip_results=False,
        save_all=save_all,
        max_msa=max_msa,
        use_cluster_profile=use_cluster_profile,
        input_features_callback=input_features_callback,
        save_recycles=save_recycles,
    )
    results_zip = f"{jobname}.result.zip"
    os.system(f"zip -r {results_zip} {jobname}")


def visualize():
    #@title Display 3D structure {run: "auto"}
    rank_num = 1 #@param ["1", "2", "3", "4", "5"] {type:"raw"}
    color = "lDDT" #@param ["chain", "lDDT", "rainbow"]
    show_sidechains = False #@param {type:"boolean"}
    show_mainchains = False #@param {type:"boolean"}

    tag = results["rank"][0][rank_num - 1]
    jobname_prefix = ".custom" if msa_mode == "custom" else ""
    pdb_filename = f"{jobname}/{jobname}{jobname_prefix}_unrelaxed_{tag}.pdb"
    pdb_file = glob.glob(pdb_filename)

    def show_pdb(rank_num=1, show_sidechains=False, show_mainchains=False, color="lDDT"):
        model_name = f"rank_{rank_num}"
        view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js',)
        view.addModel(open(pdb_file[0],'r').read(),'pdb')
        
        if color == "lDDT":
            view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
        elif color == "rainbow":
            view.setStyle({'cartoon': {'color':'spectrum'}})
        elif color == "chain":
            chains = len(queries[0][1]) + 1 if is_complex else 1
            for n,chain,color in zip(range(chains),alphabet_list,pymol_color_list):
                view.setStyle({'chain':chain},{'cartoon': {'color':color}})

        if show_sidechains:
            BB = ['C','O','N']
            view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                            {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
            view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                            {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
            view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                            {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})  
        if show_mainchains:
            BB = ['C','O','N','CA']
            view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

        view.zoomTo()
        return view

    show_pdb(rank_num, show_sidechains, show_mainchains, color).show()
    if color == "lDDT":
        plot_plddt_legend().show()
