import logging
import os
import tempfile
from pathlib import Path

import dash_bio
import dash_core_components as dcc

# Diskcache for non-production apps when developing locally
import diskcache
from dash import DiskcacheManager, html
from dash.dependencies import Input, Output, State
from dash_bio.utils import mol3dviewer_styles_creator as sparser
from dash_bio.utils import pdb_parser as parser
from flask import send_from_directory
from flask_cors import CORS
from loguru import logger

from dash_fold.layout_helper import run_standalone_app
from dash_fold.model1 import predict
from dash_fold.utils_ import PredictionCallback, make_progress_graph

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)


DATAPATH = Path(os.path.dirname(os.path.abspath(__file__))).parent / "data"


def header_colors():
    return {"bg_color": "#e7625f", "font_color": "white"}


def description():
    return (
        "Molecule visualization in 3D - perfect for viewing "
        "biomolecules such as proteins, DNA and RNA. Includes "
        "stick, cartoon, and sphere representations."
    )


def layout():
    return html.Div(
        id="mol3d-body",
        className="app-body",
        children=[
            html.Div(
                children=html.Div(
                    className="control-tab",
                    children=[
                        html.H4(className="what-is", children="What is Molecule3D?"),
                        html.P(
                            "Molecule3D hi hiis a visualizer that allows you "
                            "to view biomolecules in multiple representations: "
                            "sticks, spheres, and cartoons."
                        ),
                        html.P(
                            "You can select a preloaded structure, or upload your own, "
                            'in the "Data" tab. A sample structure is also '
                            "available to download."
                        ),
                        html.P(
                            'In the "View" tab, you can change the style and '
                            "coloring of the various components of your molecule."
                        ),
                        html.P(
                            [
                                "This app is made based on amazing google ",
                                html.A(
                                    "colab notebook",
                                    href=(
                                        "https://colab.research.google.com/github/deepmind/alphafold/"
                                        "blob/main/notebooks/AlphaFold.ipynb"
                                    ),
                                ),
                                " created by ",
                                html.A("Sergey Ovchinnikov group", href="https://twitter.com/sokrypton"),
                                ".",
                            ]
                        ),
                        html.Div(
                            [
                                html.H3("Input protein sequence(s)"),
                                dcc.Textarea(
                                    id="textarea-state-example",
                                    value="PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
                                    style={"width": "100%", "height": 200},
                                ),
                                html.P(
                                    (
                                        "Use : to specify inter-protein chainbreaks for modeling "
                                        "complexes (supports homo- and hetro-oligomers)."
                                        " For example PI...SK:PI...SK for a homodimer"
                                    ),
                                ),
                                "Job name: ",
                                dcc.Input("test", id="job-name-input", type="text"),
                                "Num relax:",
                                dcc.Dropdown([0, 1, 5], 0, id="num-relax-dropdown"),
                                html.P("specify how many of the top ranked structures to relax using amber"),
                                "template_mode:",
                                dcc.Dropdown(["none", "pdb70"], "none", id="template-mode-dropdown"),
                                html.P("none = no template information is used. pdb70 = detect templates in pdb70. "),
                            ]
                        ),
                        html.Div(
                            [
                                html.H3("MSA options (custom MSA upload, single sequence, pairing mode)"),
                                "msa_mode:",
                                dcc.Dropdown(
                                    ["mmseqs2_uniref_env", "mmseqs2_uniref", "single_sequence", "custom"],
                                    "mmseqs2_uniref",
                                    id="msa-mode-dropdown",
                                ),
                                "pair_mode:",
                                dcc.Dropdown(
                                    ["unpaired_paired", "paired", "unpaired"],
                                    "unpaired_paired",
                                    id="pair-mode-dropdown",
                                ),
                                (
                                    '"unpaired_paired" = pair sequences from same species + unpaired MSA, '
                                    '"unpaired" = seperate MSA for each chain, "paired" - only use paired sequences.'
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.H3("Advanced settings"),
                                html.Div(
                                    [
                                        "Model type:",
                                        dcc.Dropdown(
                                            [
                                                "auto",
                                                "alphafold2_ptm",
                                                "alphafold2_multimer_v1",
                                                "alphafold2_multimer_v2",
                                                "alphafold2_multimer_v3",
                                            ],
                                            "auto",
                                            id="model-type-dropdown",
                                        ),
                                        (
                                            "If auto selected, will use alphafold2_ptm for monomer prediction "
                                            "and alphafold2_multimer_v3 for complex prediction. Any of the"
                                            "mode_types can be used (regardless if input is monomer or complex)."
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        "num_recycles:",
                                        dcc.Dropdown(
                                            ["auto", "0", "1", "3", "6", "12", "24", "48"],
                                            "auto",
                                            id="num-recycles-dropdown",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        "recycle_early_stop_tolerance:",
                                        dcc.Dropdown(
                                            ["auto", 0.0, 0.5, 1.0], "auto", id="recycle-early-stop-tolerance-dropdown"
                                        ),
                                        (
                                            "if auto selected, will use 20 recycles if"
                                            " model_type=alphafold2_multimer_v3"
                                            " (with tol=0.5), all else 3 recycles (with tol=0.0)."
                                        ),
                                    ]
                                ),
                                html.H4("Sample settings"),
                                html.Ul(
                                    children=[
                                        html.Li(
                                            (
                                                "enable dropouts and increase number of seeds to sample"
                                                " predictions from uncertainty of the model."
                                            )
                                        ),
                                        html.Li("decrease max_msa to increase uncertainity"),
                                    ]
                                ),
                                html.Div(
                                    [
                                        "max_msa:",
                                        dcc.Dropdown(
                                            ["auto", "512:1024", "256:512", "64:128", "32:64", "16:32"],
                                            "auto",
                                            id="max-msa-dropdown",
                                        ),
                                    ]
                                ),
                                html.Div(["num_seeds:", dcc.Dropdown([1, 2, 4, 8, 16], 1, id="num-seeds-dropdown")]),
                                html.Div([dcc.Checklist(["Use dropout"], id="use-dropout-checklist")]),
                            ]
                        ),
                        html.Button("Submit", id="textarea-state-example-button", n_clicks=0),
                        html.Div(id="textarea-state-example-output", style={"whiteSpace": "pre-line"}),
                    ],
                )
            ),
            dcc.Loading(html.Div(id="mol3d-biomolecule-viewer", children=[])),
            html.Div(
                [
                    html.A(html.Button("Download file"), id="download-link", href="/download/alpha-proteine.pdb"),
                ]
            ),
            html.Div(
                [
                    html.P(id="paragraph_id", children=[""]),
                    dcc.Graph(
                        id="progress_bar_graph", figure=make_progress_graph(0, 10), style=dict(visibility="hidden")
                    ),
                ]
            ),
            html.Div(id="mol-sequence", children=[]),
            dcc.Store(id="mol3d-color-storage", data={}),
        ],
    )


# Function to create the modelData and style files for molecule visualization
def files_data_style(content):
    fdat = tempfile.NamedTemporaryFile(suffix=".js", delete=False, mode="w+")
    fdat.write(content)
    dataFile = fdat.name
    fdat.close()
    return dataFile


def callbacks(_app):
    @_app.callback(
        [Output("mol3d-biomolecule-viewer", "children"), Output("download-link", "href")],
        [
            Input("textarea-state-example-button", "n_clicks"),
            Input("job-name-input", "value"),
            Input("num-relax-dropdown", "value"),
            Input("template-mode-dropdown", "value"),
            Input("msa-mode-dropdown", "value"),
            Input("pair-mode-dropdown", "value"),
            Input("model-type-dropdown", "value"),
            Input("num-recycles-dropdown", "value"),
            Input("recycle-early-stop-tolerance-dropdown", "value"),
            Input("max-msa-dropdown", "value"),
            Input("num-seeds-dropdown", "value"),
            Input("use-dropout-checklist", "value"),
        ],
        [State("textarea-state-example", "value"), State("mol3d-color-storage", "data")],
        background=True,
        running=[
            (Output("textarea-state-example-button", "disabled"), True, False),
            (
                Output("paragraph_id", "style"),
                {"visibility": "hidden"},
                {"visibility": "visible"},
            ),
            (
                Output("progress_bar_graph", "style"),
                {"visibility": "visible"},
                {"visibility": "hidden"},
            ),
        ],
        progress=Output("progress_bar_graph", "figure"),
        progress_default=make_progress_graph(0, 10),
        manager=background_callback_manager,
    )
    def update_output(
        set_progress,
        n_clicks,
        job_name,
        num_relax,
        template_mode,
        msa_mode,
        pair_mode,
        model_type,
        num_recycles,
        recycle_early_stop_tol,
        max_msa,
        num_seeds,
        use_dropout,
        value,
        custom_colors,
    ):
        if n_clicks:
            # print(job_name,
            # num_relax,
            # template_mode,
            # msa_mode,
            # pair_mode,
            # model_type,
            # num_recycles,
            # recycle_early_stop_tol,
            # max_msa,
            # num_seeds,
            # use_dropout,)
            mol_style = "cartoon"
            color_style = "residue"
            # output: test 0 none mmseqs2_uniref unpaired_paired auto auto auto auto 1 None
            query = value
            job_name = f"{DATAPATH}/{job_name}"
            pred_callback = PredictionCallback(set_progress, 6)
            logger.debug("loaded parameters")
            fname = predict(
                query,
                job_name,
                num_relax,
                template_mode,
                msa_mode,
                pair_mode,
                model_type,
                num_recycles,
                recycle_early_stop_tol,
                max_msa,
                num_seeds,
                use_dropout,
                pred_callback,
            )
            logger.debug("Done prediction")
            # fname = f"{DATAPATH}/test_a5e17_6/test_a5e17_6_unrelaxed_rank_005_alphafold2_ptm_model_1_seed_000.pdb"
            fname = Path(fname)

            # Create the model data from the decoded contents
            pdb = parser.PdbParser(str(fname))
            mdata = pdb.mol3d_data()

            # Create the cartoon style from the decoded contents
            data_style = sparser.create_mol3d_style(mdata.get("atoms"), mol_style, color_style, **custom_colors)

            # Return the new molecule visualization container
            dl_link = "/download/" + "/".join(fname.parts[-2:])
            return (
                dash_bio.Molecule3dViewer(
                    id="mol-3d",
                    selectionType="atom",
                    modelData=mdata,
                    styles=data_style,
                    selectedAtomIds=[],
                    backgroundOpacity="0",
                    atomLabelsShown=False,
                ),
                dl_link,
            )
        return None, None


app = run_standalone_app(layout, callbacks, header_colors, __file__)
server = app.server
CORS(server)


log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


@server.route("/download/<path:path>")
def download(path):
    return send_from_directory(DATAPATH, path, as_attachment=True)


if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host="0.0.0.0")
