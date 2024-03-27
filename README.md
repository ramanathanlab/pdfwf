# PDFWF
Scalable PDF-to-text extraction workflow. This workflow is designed to run on
HPC systems and uses [Parsl](https://parsl-project.org/) to submit jobs to the
scheduler. The workflow supports multiple PDF parsers including
[Marker](https://github.com/VikParuchuri/marker), [Nougat](https://github.com/facebookresearch/nougat),
and **`Oreo`** (implemented here). Each parser converts PDFs to JSON lines files containing
the extracted text and optional metadata. The instructions and examples below
are for running the workflow on the [Polaris](https://www.alcf.anl.gov/polaris)
supercomputer at Argonne National Laboratory, but the workflow can be adapted
to run on other HPC systems by adding an appropriate
[Parsl configuration](https://parsl.readthedocs.io/en/stable/userguide/configuring.html).

# Installation
`pdfwf` supports several PDF parsers which have separate installation instructions below:
- [Install Marker](#marker-pipeline-installation)
- [Install Nougat](#nougat-pipeline-installation)
- [Install Oreo](#oreo-pipeline-installation)

**Note**: You may need different virtual environments for each parser.

Once you've installed your desired PDF parser, you can install `pdfwf` into your virtual environment by running:
```bash
git clone git@github.com:ramanathanlab/pdfwf.git
cd pdfwf
pip install --upgrade pip setuptools wheel
pip install -e .
```

## Usage
Running the workflow requires having at least one of these tools installed (e.g., `marker`, `nougat`, `oreo`, etc.). See [Installation](#installation) for details.

The `pdfwf` workflow can be run at scale using Parsl via the CLI as follows:
```console
> python -m pdfwf.convert --help
usage: convert.py [-h] --config CONFIG

PDF conversion workflow

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Path to workflow configuration file
```

### Workflow Configuration
The computing platform, virtual environment, parser settings, and other settings are specified via a YAML configuration file.

An example YAML file for the `pdfwf` workflow is provided below. This file can be used to run the workflow using the CLI after updating the values to be appropriate for your system.
```yaml
# The directory containing the pdfs to convert
pdf_dir: /lus/eagle/projects/argonne_tpc/hippekp/small-pdf-set

# The directory to place the converted pdfs in
out_dir: runs/output-text

# The number of PDFs to parse in each parsl task
chunk_size: 5

# The settings for the pdf parser
parser_settings:
  # The name of the parser to use
  name: marker

# The compute settings for the workflow
compute_settings:
  # The name of the compute platform to use
  name: polaris
  # The number of compute nodes to use
  num_nodes: 1
  # Make sure to update the path to your conda environment and HF cache
  worker_init: "module load conda/2023-10-04; conda activate marker-wf; export HF_HOME=<path-to-your-HF-cache-dir>"
  # The scheduler options to use when submitting jobs
  scheduler_options: "#PBS -l filesystems=home:eagle:grand"
  # Make sure to change the account to the account you want to charge
  account: <your-account-name-to-charge>
  # The HPC queue to submit to
  queue: debug
  # The amount of time to request for your job
  walltime: 01:00:00
```

We provide example configurations for each parser in these files:
- **marker**: [examples/marker/marker_test.yaml](examples/marker/marker_test.yaml)
- **nougat**: [examples/nougat/nougat_test.yaml](examples/nougat/nougat_test.yaml)
- **oreo**: [examples/oreo/oreo_test.yaml](examples/oreo/oreo_test.yaml)

**Note**: Please see the comments in the example YAML files for **documentation
 on the settings**.

### Running the Workflow
Once you've updated the YAML file with your environment, project information, and file paths
the workflow can be run directly from a login node using the CLI as follows:
```
nohup python -m pdfwf.convert --config <your-config.yaml> &> nohup.out &
```

The output of the workflow will be written to the `out_dir` specified in the
configuration file. The `<out_dir>/parsed_pdfs` directory will contain the
parsed PDFs in JSON lines format. Each line of the JSONL file will contain
(at least) a "text" field containing the parsed text for a given PDF and a
"path" field containing the path to the PDF. See the example below:
```json
{"path": "/path/to/1.pdf", "text": "This is the text of the first PDF."}
{"path": "/path/to/2.pdf", "text": "This is the text of the second PDF."}
```

**Note**: If the parser fails to parse a PDF, the JSONL file will not contain
an entry for that PDF.

See the [Monitoring the Workflow](#monitoring-the-workflow) section for
description of the other log files that are generated during the workflow.

### Monitoring the Workflow
Once you've started the workflow, you can monitor the outputs by watching the
output files in the `out_dir` specified in the configuration file. Below are
some useful commands to monitor the workflow. First, `cd` to the `out_dir`.

To see the number of PDFs that have been parsed:
```console
cat parsed_pdfs/* | grep '{"path":' | wc -l
```

To watch the stdout and stderr of the tasks:
```console
tail -f parsl/000/submit_scripts/*
```

To check the Parsl workflow log:
```console
tail -f parsl/000/parsl.log
```

To see the basic workflow log:
```console
cat pdfwf.log
```

### Stopping the Workflow
If you'd like to stop the workflow while it's running, you need to
stop the Python process, the Parsl high-throughput executor process, and then `qdel` the job ID.
The process IDs can be determined using the `ps` command. The job ID can be found using `qstat`.

## `Marker` Pipeline Installation

This setup will install the `Marker` tool in a new environment.

On a compute node of Polaris, follow the following instructions:

_Note setting up the local.env assumes conda, see [Non-Conda Env](#non-conda-env)_
```bash
mkdir wf-validation
cd wf-validation/

# Create a base conda environment
module load conda/2023-10-04
conda create -n marker-wf python=3.10
conda activate marker-wf

# Install Marker
git clone https://github.com/VikParuchuri/marker.git
cd marker/
conda install -c conda-forge tesseract -y
conda install -c conda-forge ghostscript -y
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
python3 -m pip install pip setuptools wheel transformers deepspeed torch==2.0.1+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --force-reinstall --upgrade -vvv

##### Setup local.env for Marker #####

# Option 1: manually create the file. Must have the conda environment open and tesseract installed
touch local.env
echo "TESSDATA_PREFIX=$(find $CONDA_PREFIX -name tessdata)" > local.env
echo "TORCH_DEVICE=cuda" >> local.env
echo "INFERENCE_RAM=40" >> local.env

# Option 2: Use the script provided in utils
bash $path_to_pdfwf_root/utils/setup_marker.sh local.env

#### END Setup local.env ####
```

### Non-Conda Env
If you are not using conda, these instructions will establish the local.env file
```bash
# Replace the path below with the path to your environment/where you install tesseract
find /home/hippekp/CVD-Mol_AI/hippekp/conda/envs/marker-wf/ -name tessdata
# replace the path in this command with the output of the above command
echo "TESSDATA_PREFIX=/home/hippekp/CVD-Mol_AI/hippekp/conda/envs/marker-wf/share/tessdata" >> marker/local.env
echo "TORCH_DEVICE=cuda" >> marker/local.env
echo "INFERENCE_RAM=40" >> marker/local.env
```

## `Nougat` Pipeline Installation
On a compute node, run:
```bash
# Create a conda environment with python3.10
module load conda/2023-10-04
conda create -n nougat-wf python=3.10
conda activate nougat-wf

# Install Nougat
python3 -m pip install --upgrade pip setuptools wheel chardet
pip install -r requirements/nougat_requirements.txt

# Note: If your system has CUDA 12.1, Nougat environment installation should now be complete. At the time of writing, Polaris uses CUDA 11.8. So, install the right torch binary using the command below.
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

This should conclude the installation of Nougat. Now, get a Polaris *compute node* to confirm it runs successfully in your environment.

```
# Copy a sample pdf file to an arbitrary location.

# Run Nougat inference on the pdfs with the command below.

nougat /path/to/your_file.pdf -o /path/to/output_dir -m 0.1.0-base -b 10

# This command should write a your_file.mmd file into your output directory which will be automatically created if it doesn't exist.
```
If `Nougat` inference ran successfully, proceed to install the `pdfwf` workflow using the instructions in [Installation](#installation).

## `Oreo` Pipeline Installation
On a compute node, run:
```bash
module load conda/2023-10-04
conda create -n pdfwf python=3.10 -y
conda activate pdfwf
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements/oreo_requirements.txt
```

## CLI
For running smaller jobs without using the Parsl workflow, the CLI can be used.
The CLI provides different commands for running the various parsers. The CLI
can run the `marker`, `nougat`, and `oreo` parsers. The CLI does not submit jobs
to the scheduler and is intended for use on small datasets that can be
processed on a single interactive node or workstation.

**Usage**:

```console
$ [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `marker`: Parse PDFs using the marker parser.
* `nougat`: Parse PDFs using the nougat parser.
* `oreo`: Parse PDFs using the oreo parser.

## `marker`

Parse PDFs using the marker parser.

**Usage**:

```console
$ pdfwf marker [OPTIONS]
```

**Options**:

* `-p, --pdf_path PATH`: The directory containing the PDF files to convert (recursive glob).  [required]
* `-o, --output_dir PATH`: The directory to write the output JSON lines file to.  [required]
  * `-nc, --num_conversions INTEGER`: Number of pdfs to convert (useful for debugging, by default convert every document) [default: 0].
* `--help`: Show this message and exit.

## `nougat`

Parse PDFs using the nougat parser.

**Usage**:

```console
$ pdfwf nougat [OPTIONS]
```

**Options**:

* `-p, --pdf_path PATH`: The directory containing the PDF files to convert (recursive glob).  [required]
* `-o, --output_dir PATH`: The directory to write the output JSON lines file to.  [required]
* `-bs, --batchsize INTEGER`: Number of pages per patch. Maximum 10 for A100 40GB.  [default: 10]
* `-c, --checkpoint PATH`: Path to existing or new Nougat model checkpoint  (to be downloaded)  [default: nougat_ckpts/base]
* `-m, --mmd_out PATH`: The directory to write optional mmd outputs along with jsonls.
* `-r, --recompute`: Override pre-existing parsed outputs.
* `-f, --full_precision`: Use float32 instead of bfloat32.
* `-md, --markdown`: Output pdf content in markdown compatible format.  [default: True]
* `-s, --skipping`: Skip if the model falls in repetition.  [default: True]
* `-n, --nougat_logs_path PATH`: The path to the Nougat-specific logs.  [default: pdfwf_nougat_logs]
* `-nc, --num_conversions INTEGER`: Number of pdfs to convert (useful for debugging, by default convert every document) [default: 0].
* `--help`: Show this message and exit.

## `oreo`

Parse PDFs using the oreo parser.

**Usage**:

```console
$ pdfwf oreo [OPTIONS]
```

**Options**:

* `-p, --pdf_path PATH`: The directory containing the PDF files to convert (recursive glob).  [required]
* `-o, --output_dir PATH`: The directory to write the output JSON lines file to.  [required]
* `-d, --detection_weights_path PATH`: Weights to layout detection model.
* `-t, --text_cls_weights_path PATH`: Model weights for (meta) text classifier.
* `-s, --spv05_category_file_path PATH`: Path to the SPV05 category file.
* `-d, --detect_only`: File type to be parsed (ignores other files in the input_dir)
* `-m, --meta_only`: Only parse PDFs for meta data
* `-e, --equation`: Include equations into the text categories
* `-t, --table`: Include table visualizations (will be stored)
* `-f, --figure`: Include figure  (will be stored)
* `-s, --secondary_meta`: Include secondary meta data (footnote, headers)
* `-a, --accelerate`: If true, accelerate inference by packing non-meta text patches
* `-b, --batch_yolo INTEGER`: Main batch size for detection/# of images loaded per batch  [default: 128]
* `-v, --batch_vit INTEGER`: Batch size of pre-processed patches for ViT pseudo-OCR inference  [default: 512]
* `-c, --batch_cls INTEGER`: Batch size K for subsequent text processing  [default: 512]
* `-x, --bbox_offset INTEGER`: Number of pixels along which  [default: 2]
* `-nc, --num_conversions INTEGER`: Number of pdfs to convert (useful for debugging, by default convert every document) [default: 0].
* `--help`: Show this message and exit.


## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```

To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```

To generate the CLI documentation, run:
```
typer pdfwf.cli utils docs --output CLI.md
```
