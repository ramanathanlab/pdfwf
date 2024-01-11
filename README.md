# PDFWF
PDF-to-text extraction workflow.

## Whole Pipeline Installation 

One a compute node of polaris, follow the following instructions: 

_Note the commented lines where you have to change to paths that fit your directories._
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

# Setup local.env for Marker
touch marker/local.env
# replace the path below with the path to your conda environment
find /home/hippekp/CVD-Mol_AI/hippekp/conda/envs/marker-wf/ -name tessdata
# replace the path in this command with the output of the above command
echo "TESSDATA_PREFIX=/home/hippekp/CVD-Mol_AI/hippekp/conda/envs/marker-wf/share/tessdata" >> marker/local.env
echo "TORCH_DEVICE=cuda" >> marker/local.env
echo "INFERENCE_RAM=40" >> marker/local.env

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

## `pdfwf` _Only_ Installation

Create environment using preferred method. If locally/not on HPC:

```
conda create -n pdfwf python=3.10
```
If on HPC I recomend [installing Marker](#tool-installation) first then installing this package in the marker environment as it has less dependencies.

Install from the github root 
```
pip install -e .
```

## Usage 
Requires having the tool (e.g `marker`, `nougat` etc.) installed. See [Tool installation](#tool-installation) for more details.

```
> python -m pdfwf.convert --help 
usage: convert.py [-h] [--pdf-dir PDF_DIR] [--out-dir OUT_DIR] [--run-dir RUN_DIR] [--hf-cache HF_CACHE] [--num-nodes NUM_NODES]
                  --account ACCOUNT [--queue QUEUE] [--walltime WALLTIME] [--num_conversions NUM_CONVERSIONS]

options:
  -h, --help            show this help message and exit
  --pdf-dir PDF_DIR     Directory containing pdfs to convert
  --out-dir OUT_DIR     Directory to place converted pdfs in
  --run-dir RUN_DIR     Directory to place parsl run files in
  --hf-cache HF_CACHE   Directory to place marker huggingface cache in
  --num-nodes NUM_NODES
                        Number of nodes to use for conversion
  --account ACCOUNT     Account to charge for job
  --queue QUEUE         Queue to use on polaris
  --walltime WALLTIME   Max walltime for job in form HH:MM:SS
  --num_conversions NUM_CONVERSIONS
                        Number of pdfs to convert (useful for debugging)

```

Example command: 
```
python -m pdfwf.convert --pdf-dir pdf-dir --out-dir output-md --run-dir parsl --hf-cache hf-cache-dir --num-nodes 20 --queue p
rod --walltime 03:00:00 --account account-name
```

## Tool installation 

### Marker 

[Marker Github](https://github.com/VikParuchuri/marker)

__Installation Instructions (*Polaris*)__
I would clone the base conda environment on Polaris and then install `pdfwf` in that environment. Cloning the base environment can be done with the following command:

```
conda create -n marker --clone base
```
These commands install the dependencies for `marker` and make the :

```
conda install -c conda-forge tesseract
conda install -c conda-forge ghostscript

# Setup local.env in `marker_root/marker/` directory
touch marker/local.env
# Point the find command to wherever your conda root is 
# Add location to `local.env` 
find ~/miniconda3  -name tessdata


python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
python3 -m pip install pip setuptools wheel transformers deepspeed torch==2.0.1+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --force-reinstall --upgrade -vvv

```

*Note*: The local.env file is installation dependent. See [Marker Usage](https://github.com/VikParuchuri/marker?tab=readme-ov-file#usage) for more details on how to set it up. Here is an example of what the `local.env` looks like:

```
TESSDATA_PREFIX=~/conda/envs/marker/share/tessdata
TORCH_DEVICE=cuda
INFERENCE_RAM=40
```

Example for running `marker` from the original codebase: 
```
python convert_single.py in.pdf out.md
```
