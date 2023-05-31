# DashFold -- port of [Colabfold](https://github.com/sokrypton/ColabFold) to Dash

We took Colabfold and rewrote it with the Dash framework as a proof of concept for SciLife Serve web app.

## Installation

Refer to [docker/Dockerfile](docker/Dockerfile) in order to install this app.

## Usage

This repo is packeged into a docker image which could be found under `kuparez/dashfold` on docker hub.

## Limitations

- It's really slow. On a 32 core CPU it takes about 40 minutes to finish predictions. It's really slow for a browser app and the one that you cannot close or disconnect from;
  - Potential solution would be to use a queue based system for this kind of compute jobs.
- It's ugly;
- Has some dirty hacks to work. Not recommended for production use. These are solutions that I don't really like:
  - No CSRF token;
  - Without it, [this part](https://github.com/ScilifelabDataCentre/dashfold/blob/916f6e0bfd99270af80ebf3b1524af5840d7bc38/dash_fold/app.py#L351) is dangerous https://blog.vnaik.com/posts/web-attacks.html;
  - Uses [on-disk brocker](https://github.com/ScilifelabDataCentre/dashfold/blob/916f6e0bfd99270af80ebf3b1524af5840d7bc38/dash_fold/app.py#L24) for incoming tasks;

# References

- https://github.com/mpg-age-bioinformatics/flaski/blob/4c3a2b88819199e73563d853f772259052acaf93/routes/apps/alphafold.py
- https://huggingface.co/spaces/osanseviero/esmfold
- https://esmfold.streamlit.app

