# AdaParse
AdaParse (**Ada**ptive Parallel **P**DF Parsing **a**nd **R**esource **S**caling **E**ngine) enable scalable high-accuracy PDF parsing. AdaParse is a data-driven strategy that assigns an appropriate parser to each document; offering high accuracy for any computaional budget.
Moreover, it offers a workflow of various PDF parsing software that includes
- extraction tools: [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/), [pypdf](https://pypdf.readthedocs.io/en/stable/)
- traditional OCR: [Tesseract](https://github.com/tesseract-ocr/tesseract),
- modern OCR (e.g., Vision Transformers): [Nougat](https://github.com/facebookresearch/nougat), and [Marker](https://github.com/VikParuchuri/marker)

AdaParse designed to run on HPC systems and has parsed millions of (scientific) PDFs. It uses [Parsl](https://parsl-project.org/) to submit jobs to the
scheduler. While AdaParse is agnostic to the specific system, instructions below are tailored to the [Polaris](https://www.alcf.anl.gov/polaris)
supercomputer at Argonne National Laboratory (ANL). Regardless, AdaParse can run on any system (large or small) by adding an appropriate
[Parsl configuration](https://parsl.readthedocs.io/en/stable/userguide/configuring.html).

# Installation
The steps below enable any of the parsers.
```bash
git clone git@github.com:7shoe/AdaParse.git
cd AdaParse
pip install --upgrade pip setuptools wheel
pip install -e .
```
If you plan on using [Tesseract](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract), additional installation steps are required.

## Usage
The `adaparse` workflow can be run at scale using Parsl
```console
> python -m adaparse.convert --help
usage: convert.py [-h] --config CONFIG

PDF conversion workflow

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Path to workflow configuration file
```
A single command triggers the embarassingly parallel PDF parsing engine:
```
python -m adaparse.convert --config <your-config.yaml>
```

### Configuration
The YAML configuration file specifies all aspects of the chosen parser, virtual environment and computing platform it is run on.

An sample configuration YAML file is provided below.
```yaml
# The directory containing the PDFs to be parsed
pdf_dir: /lus/eagle/projects/argonne_tpc/siebenschuh/small-pdf-dataset

# The directory to store the JSONLs
out_dir: runs/output-dir

# The number of PDFs per parsl task
chunk_size: 5

# Parser settings
parser_settings:
  # The name of the parser to use
  name: adaparse

# Compute settings (e.g., ANL's Polaris)
compute_settings:
  # The name of the compute platform to use
  name: polaris
  # The number of compute nodes to use
  num_nodes: 1
  # Activate conda environment and set HF cache path
  worker_init: "module use /soft/modulefiles; module load conda/2024-04-29; conda activate adaparse; export HF_HOME=<path-to-your-HF-cache-dir>"
  # Scheduler options
  scheduler_options: "#PBS -l filesystems=home:eagle"
  # Your account/project that will be charged
  account: <your-account-name-to-charge>
  # The HPC queue to submit to
  queue: debug
  # The amount of runtime requested for this job
  walltime: 01:00:00
```
Example configuration files for each parser can be found in:
- **AdaParse**: [examples/adaparse/adaparse_test.yaml](examples/adaparse/adaparse_test.yaml)
- **Nougat**: [examples/nougat/nougat_test.yaml](examples/nougat/nougat_test.yaml)
- **Marker**: [examples/marker/marker_test.yaml](examples/marker/marker_test.yaml)
- **PyMuPDF**: [examples/pymupdf/pymupdf_test.yaml](examples/pymupdf/pymupdf_test.yaml)
- **pypdf**: [examples/pymupdf/pymupdf_test.yaml](examples/pymupdf/pymupdf_test.yaml)
- **Tesseract**: [examples/tesseract/tesseract_test.yaml](examples/tesseract/tesseract_test.yaml)

### Output
Once you've updated the YAML file and run the AdaParse command, the textual output will be written to the `out_dir`.
The subdirectory `<out_dir>/parsed_pdfs` contains the parsed PDF output in JSON lines format. Each line of the JSONL file contains
the `path` field with the PDF source file, the `text` field containing the parsed text, and the `metadata` field containing information on author, title. etc..
Please note that the particular metadata stored depends on the parser used.
```json
{"path": "/path/to/1.pdf", "text": "This is the text of the first PDF."}
{"path": "/path/to/2.pdf", "text": "This is the text of the second PDF."}
```

**Note**: If the parser fails to parse a PDF, the JSONL file will not contain
an entry for that PDF.

See the [Monitoring the Workflow](#monitoring-the-workflow) section for
description of the other log files that are generated during the workflow.


## Developement
It is recommended to use a virtual environment for developement. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
