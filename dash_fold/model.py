# @title Input protein sequence(s), then hit `Runtime` -> `Run all`
import glob
import hashlib
import os
import re
import sys
import warnings
from pathlib import Path
from sys import version_info

import matplotlib.pyplot as plt
from Bio import BiopythonDeprecationWarning
from colabfold.batch import get_queries, run, set_model_type
from colabfold.download import download_alphafold_params
from colabfold.plot import plot_msa_v2
from colabfold.utils import setup_logging
from loguru import logger

PYTHON_VERSION = f"{version_info.major}.{version_info.minor}"
warnings.simplefilter(action="ignore", category=BiopythonDeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def add_hash(x, y):
    return x + "_" + hashlib.sha1(y.encode()).hexdigest()[:5]


def predict(
    query_sequence,
    jobname,
    num_relax,
    template_mode,
    msa_mode,
    pair_mode,
    model_type,
    num_recycles,
    recycle_early_stop_tolerance,
    max_msa,
    num_seeds,
    use_dropout,
    pred_callback,
):

    use_amber = num_relax > 0

    # remove whitespaces
    query_sequence = "".join(query_sequence.split())

    basejobname = "".join(jobname.split())
    # basejobname = re.sub(r'\W+', '', basejobname)
    basejobname_1 = re.sub(r"\W+", "", basejobname)
    print(f"basejobname_1 = {basejobname_1}")
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
        custom_template_path = os.path.join(jobname, "template")
        os.makedirs(custom_template_path, exist_ok=True)
        use_templates = True
    else:
        custom_template_path = None
        use_templates = False

    print("jobname", jobname)
    print("sequence", query_sequence)
    print("length", len(query_sequence.replace(":", "")))

    # @markdown ### MSA options (custom MSA upload, single sequence, pairing mode)
    # @markdown - "unpaired_paired" = pair sequences from same species + unpaired MSA,
    # "unpaired" = seperate MSA for each chain, "paired" - only use paired sequences.

    # decide which a3m to use
    if "mmseqs2" in msa_mode:
        a3m_file = os.path.join(jobname, f"{jobname}.a3m")
    else:
        a3m_file = os.path.join(jobname, f"{jobname}.single_sequence.a3m")
        with open(a3m_file, "w") as text_file:
            text_file.write(">1\n%s" % query_sequence)

    use_dropout = use_dropout is not None

    num_recycles = None if num_recycles == "auto" else int(num_recycles)
    recycle_early_stop_tolerance = (
        None if recycle_early_stop_tolerance == "auto" else float(recycle_early_stop_tolerance)
    )
    if max_msa == "auto":
        max_msa = None

    dpi = 200  # @param {type:"integer"}
    # @markdown - set dpi for image resolution

    # @markdown Don't forget to hit `Runtime` -> `Run all` after updating the form.

    # @title Run Prediction
    display_images = True  # @param {type:"boolean"}

    # For some reason we need that to get pdbfixer to import
    if use_amber and f"/usr/local/lib/python{PYTHON_VERSION}/site-packages/" not in sys.path:
        sys.path.insert(0, f"/usr/local/lib/python{PYTHON_VERSION}/site-packages/")

    def input_features_callback(input_features):
        if display_images:
            plot_msa_v2(input_features)
            plt.show()
            plt.close()

    result_dir = jobname
    if "logging_setup" not in globals():
        setup_logging(Path(os.path.join(jobname, "log.txt")))

    queries, is_complex = get_queries(queries_path)
    model_type = set_model_type(is_complex, model_type)

    if "multimer" in model_type and max_msa is not None:
        use_cluster_profile = False
    else:
        use_cluster_profile = True

    download_alphafold_params(model_type, Path("."))
    pred_callback.update_progress_bar()
    logger.info("Beginning internal prediction")
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
        model_order=[1, 2, 3, 4, 5],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        rank_by="auto",
        pair_mode=pair_mode,
        stop_at_score=float(100),
        prediction_callback=pred_callback,
        dpi=dpi,
        zip_results=False,
        save_all="save_all",
        max_msa=max_msa,
        use_cluster_profile=use_cluster_profile,
        input_features_callback=input_features_callback,
        save_recycles="save_recycles",
    )
    results_zip = f"{jobname}.result.zip"
    os.system(f"zip -r {results_zip} {jobname}")

    # @title Display 3D structure {run: "auto"}

    rank_num = 1  # @param ["1", "2", "3", "4", "5"] {type:"raw"}

    tag = results["rank"][0][rank_num - 1]
    jobname_prefix = ".custom" if msa_mode == "custom" else ""
    pdb_filename = f"{jobname}/{jobname.replace('/', '_')}{jobname_prefix}_unrelaxed_{tag}.pdb"
    pdb_file = glob.glob(pdb_filename)

    return pdb_file[0]
