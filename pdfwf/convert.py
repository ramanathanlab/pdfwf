from pathlib import Path
from argparse import ArgumentParser

import parsl 
from parsl import python_app 

from pdfwf.config import get_config

@python_app
def torch_test_app(): 
    import torch 
    import os 
    return torch.cuda.is_available(), os.environ['CUDA_VISIBLE_DEVICES']

@python_app
def marker_single_app(pdf_path: str, out_path: str): 
    import json
    import os
    from marker.models import load_all_models
    from marker.convert import convert_single_pdf
    from pathlib import Path

    pdf_name = Path(pdf_path).stem

    model_lst = load_all_models()
    full_text, out_meta = convert_single_pdf(pdf_path, model_lst)

    output_md = os.path.join(out_path, pdf_name + ".md")
    with open(output_md, "w+", encoding='utf-8') as f:
        f.write(full_text)

    out_meta_filename = os.path.join(out_path, pdf_name + ".metadata.json")
    with open(out_meta_filename, "w+", encoding='utf-8') as f:
        f.write(json.dumps(out_meta, indent=4))

    return output_md, out_meta_filename


if __name__ == "__main__":

    
    parser = ArgumentParser()
    parser.add_argument("--run_dir", default="./parsl", type=Path, help="Directory to place parsl run files in")


    args = parser.parse_args()

    run_dir = str(args.run_dir.resolve())

    user_opts = {
        "run_dir":          run_dir,
        "worker_init":      f"module load conda/2023-10-04;conda activate marker; cd {run_dir}", # load the environment where parsl is installed
        "scheduler_options":"#PBS -l filesystems=home:eagle:grand" , # specify any PBS options here, like filesystems
        "account":          "RL-fold",
        "queue":            "debug",
        "walltime":         "1:00:00",
        "nodes_per_block":  1, # number of nodes to allocate
        "cpus_per_node":    32, # Up to 64 with multithreading
        "available_accelerators": 4, # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
        "cores_per_worker": 8, # this will set the number of cpu hardware threads per worker.  
    }
    
    config = get_config(user_opts)
    parsl.load(config) 

    # Test Marker text extraction 
    fp = "/lus/eagle/projects/radbio/papers/0000bad036a5fa1391c8fbcc291a92f86b577b01.pdf"
    out_path = str(Path("text_out").resolve())
    Path(out_path).mkdir(exist_ok=True, parents=True)

    print(marker_single_app(fp, out_path).result())



