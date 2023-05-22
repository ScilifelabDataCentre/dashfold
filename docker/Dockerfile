FROM continuumio/miniconda3

RUN apt update && apt -y upgrade && apt install -y git wget zip gcc g++ clang

COPY environment.yaml .
RUN conda env update -f environment.yaml
RUN conda clean -y -a
RUN pip install --no-cache-dir "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold" biopython dash
RUN pip install --no-cache-dir --upgrade jax==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
COPY . /proj
WORKDIR /proj
RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir openpyxl
ENV GUNICORN_CMD_ARGS "--bind=0.0.0.0:8000 --workers=1 --thread=1 --worker-class=gthread --forwarded-allow-ips='*'"
CMD ["gunicorn", "dash_fold.app:server"]
