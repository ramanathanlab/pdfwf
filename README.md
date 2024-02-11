# PDFWF
PDF-to-text extraction workflow.

# Installation
There are a number of ways to install this software. If you already have an environment with Marker installed, please see [`pdfwf` only installation](#pdfwf-only-installation). If you do not have Marker installed, please see [Marker Pipeline Installation](#marker-pipeline-installation) for instructions on how to install the whole pipeline with Marker.

## Marker Pipeline Installation

This setup will install both the `Marker` tool and the `pdfwf` workflow from a new environment.

One a compute node of polaris, follow the following instructions:

_Note setting up the local.env assumes conda, see [Non-Conda Env](#non-conda-env)_
```
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

# Test the installation on a PDF
python convert_single.py /lus/eagle/projects/argonne_tpc/TextCollections/OSTI/RawData/Journal_Article/1333379.pdf ../1333379.md
```

Once you verify that marker works on the example given, exit the compute node and from a login node execute the following:

_Note the line where you must change a file to match your conda environment_
```
cd wf-validation
# Activate your environment with marker and install the workflow package
module load conda/2023-10-04
conda activate marker-wf
git clone https://github.com/ramanathanlab/pdfwf.git
cd pdfwf/
pip install -e .
# Insert your environment into the init, working on a parameterized solution currently.
vim pdfwf/convert.py # edit line 70 to point to your conda environment name, e.g 'conda activate marker-wf' instead of 'conda activate marker'

# Run the workflow on a small set of 10 pdfs
python -m pdfwf.convert --pdf-dir /lus/eagle/projects/argonne_tpc/hippekp/small-pdf-set --out-dir ../small-pdf-text --run-dir ../parsl --num-nodes 2 --queue debug --walltime 01:00:00 --account [ACCOUNT]
```

#### Non-Conda Env
If you are not using conda, these instructions will establish the local.env file
```
# replace the path below with the path to your environment/where you install tesseract
find /home/hippekp/CVD-Mol_AI/hippekp/conda/envs/marker-wf/ -name tessdata
# replace the path in this command with the output of the above command
echo "TESSDATA_PREFIX=/home/hippekp/CVD-Mol_AI/hippekp/conda/envs/marker-wf/share/tessdata" >> marker/local.env
echo "TORCH_DEVICE=cuda" >> marker/local.env
echo "INFERENCE_RAM=40" >> marker/local.env

```

## `pdfwf` _Only_ Installation

This assumes that the environment you are currently using has the `Marker` tool installed. If you do not have `Marker` installed, please see [Whole Pipeline Installation](#whole-pipeline-installation) for instructions on how to install the whole pipeline with Marker.

From this repositories root
```
pip install --upgrade pip setuptools wheel
pip install -e .
```

To install the dependencies for the Oreo parser
```
pip install -r requirements/oreo_requirements.txt
```

For development, you can install the devtools using:
```
pip install -e '.[dev,docs]'
```

## Usage
Requires having the tool (e.g `marker`, `nougat` etc.) installed. See [Tool installation](#tool-installation) for more details.

The `pdfwf` workflow can be run using the CLI as follows:
```
> python -m pdfwf.convert --help
usage: convert.py [-h] --config CONFIG

PDF conversion workflow

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Path to workflow configuration file
```

An example YAML file for the `pdfwf` workflow is provided below. This file can be used to run the workflow using the CLI after replacing the values in the file with the appropriate values for your system.
```yaml
# The directory containing the pdfs to convert
pdf_dir: /lus/eagle/projects/argonne_tpc/hippekp/small-pdf-set

# The directory to place the converted pdfs in
out_dir: output-text

# The compute settings for the workflow
compute_settings:
  name: polaris
  num_nodes: 20
  # Make sure to update the path to your conda environment and HF cache
  worker_init: "module load conda/2023-10-04; conda activate marker-wf; export HF_HOME=<path-to-your-HF-cache-dir>"
  scheduler_options: "#PBS -l filesystems=home:eagle:grand"
  # Make sure to change the account to the account you want to charge
  account: <your-account-name-to-charge>
  queue: prod
  walltime: 03:00:00
```

For example, the workflow can be run using the CLI as follows:
```
python -m pdfwf.convert --config config.yaml
```
