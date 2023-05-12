import os
import tempfile
from datetime import datetime

import plotly.graph_objects as go
from loguru import logger


def timestamp():
  now = datetime.now()
  dt_string = now.strftime("%Y%m%d.%H%M%S.")
  return dt_string


def make_submission_file(suffix,folder="submissions"):
    dt_string=timestamp()
    new_file, filename = tempfile.mkstemp(suffix=suffix, prefix=dt_string )
    os.close(new_file)
    if not os.path.isdir(f"/{folder}/"):
        os.makedirs(f"/{folder}/")
    filename=f"/{folder}/"+os.path.basename(filename)
    return filename


def make_options(valuesin):
    opts=[]
    for c in valuesin:
        opts.append( {"label":c, "value":c} )
    return opts


def make_progress_graph(progress, total):
    logger.debug("Updating progress bar")
    progress_graph = (
        go.Figure(data=[go.Bar(x=[progress])])
        .update_xaxes(range=[0, total])
        .update_yaxes(
            showticklabels=False,
        )
        .update_layout(height=100, margin=dict(t=20, b=40))
    )
    return progress_graph


class PredictionCallback:

    def __init__(self, set_progress_fun, total_count):
        self.total_count = total_count
        self.set_progress_fun = set_progress_fun
        self.count = 0

    def draw_protein(self, protein_obj, length,
                            prediction_result, input_features, mode):
        model_name, relaxed = mode
        if not relaxed:
            fig = plot_protein(protein_obj, Ls=length, dpi=150)
            plt.show()
            plt.close()

    def __call__(self, protein_obj, length,
                            prediction_result, input_features, mode):
        # self.draw_protein(protein_obj, length, prediction_result, input_features, mode)
        self.update_progress_bar()

    def update_progress_bar(self):
        self.count += 1
        self.set_progress_fun(make_progress_graph(self.count, self.total_count))
