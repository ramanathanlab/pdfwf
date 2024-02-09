from __future__ import annotations

from tensor_utils import *

import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer   # pipeline
from pylatexenc.latex2text import LatexNodes2Text

from texify2.texify.inference import accelerated_batch_inference
from texify2.texify.model.model import load_model
from texify2.texify.model.processor import load_processor
from texify2.texify.settings import settings


def main():
    # parser
    parser = argparse.ArgumentParser(description="Process command line arguments")
    
    # paths
    parser.add_argument("-in", "--input_dir", type=str, help="Input directory path from which document files are sourced.")
    parser.add_argument("-out", "--output_dir", type=str, help="Output path into which json/visuals are stored.")
    parser.add_argument("-f", "--file_type", type=str, default='pdf', help="File type to be parsed (ignores other files in the input_dir).")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which majority of data processing occurs.")
    
    # boolean arguments (subsetting relevant classes)
    parser.add_argument("--detect_only", action="store_true", help="Only scan PDFs for meta statistics on its attributes.")
    parser.add_argument("-mo",  "--meta_only", action="store_true", help="Only parse PDFs for meta data")
    parser.add_argument("-eq",  "--equation", action="store_true", help="Include equations into the text categories")
    parser.add_argument("-tab", "--table", action="store_true", help="Include table visualizations (will be stored).")
    parser.add_argument("-fig", "--figure", action="store_true", help="Include figure  (will be stored).")
    parser.add_argument("-sec", "--secondary_meta", action="store_true", help="Include secondary meta data (footnote, headers).")
    parser.add_argument("-a", "--accelerate", action="store_true", help="If true, accelerate inference by packing non-meta text patches.")

    # batch sizes (tuned for single-GPU performance)
    parser.add_argument("--batch_yolo", type=int, default=128, help="Main batch size for detection/# of images loaded per batch.")
    parser.add_argument("--batch_vit", type=int, default=512, help="Batch size N for number of pre-processed patches for ViT pseudo-OCR inference.")
    parser.add_argument("--batch_cls", type=int, default=512, help="Batch size K for subsequent text processing.")
    parser.add_argument("--max_page_to_store_visuals", type=int, default=1000, help="Max. number of pages in the dset for which table/figure saving is allowed.")
    
    # finetuning parameters
    parser.add_argument("--bbox_offset", type=int, default=2, help="Number of pixels along which")
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float16, help="Dtype of ViT OCR model.")
    
    # model weights
    parser.add_argument("--detection_weights", 
                        type=str,
                        default="./yolov5/runs/train/best_SPv05_run/weights/best.pt",
                        help="Weights to layout detection model.")
    parser.add_argument("--text_cls_weights", 
                        type=str,
                        default="./text_classifier/meta_text_classifier", 
                        help="Model weights for (meta) text classifier.")
    
    # parse arguments
    args = parser.parse_args()
    
    # directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    file_type  = args.file_type
    device   = args.device
    
    # - check validity
    assert os.path.isdir(input_dir),  f"Input directory is invalid; does not exist: `input_dir`={input_dir}"
    assert os.path.isdir(output_dir), f"Output directory is invalid; does not exist: `output_dir`={output_dir}"
    assert device in ['cuda', 'cpu'], "Chosen `device` must be either `cuda` or `cpu`."
    
    # device availability
    if(device=="cude" and not(torch.cuda.is_available())):
        print("--device `cude` was chosen but is not available. CPU-processing only.")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # models
    detection_weights_path = args.detection_weights
    txt_cls_weights_path   = args.text_cls_weights
    assert os.path.isfile(detection_weights_path), f"Path to weights of detection model does not exist: {detection_weights_path}"
    assert os.path.isdir(txt_cls_weights_path), f"Path to weights of text classification model does not exist {txt_cls_weights_path}"

    # flags
    detect_only = args.detect_only
    meta_only   = args.meta_only
    equation_flag = args.equation
    table_flag = args.table
    fig_flag = args.figure
    secondary_meta  = args.secondary_meta
    accelerate_flag = args.accelerate
    
    # numbers
    max_page_to_store_visuals = args.max_page_to_store_visuals
    offset = args.bbox_offset
    
    
    # batch sizes
    batch_yolo = args.batch_yolo
    batch_vit = args.batch_vit
    batch_cls = args.batch_cls
    
    # check: `exclusive` flag
    if detect_only and (equation_flag or table_flag or fig_flag or secondary_meta or meta_only):
        parser.error("The `--detect_only` flag cannot be used with any other flag.")
    if meta_only and (equation_flag or table_flag or fig_flag or secondary_meta or detect_only):
        parser.error("The `--meta_only` flag cannot be used with any other flag.")
        
    
    # identify relevant classes and group by treatment
    rel_txt_classes      = get_relevant_text_classes('pdf', meta_only, equation_flag, table_flag, fig_flag, secondary_meta)
    rel_meta_txt_classes = get_relevant_text_classes('pdf', meta_only=True)
    rel_visual_classes   = get_relevant_visual_classes('pdf', table_flag=table_flag, fig_flag=fig_flag)
    unpackable_classes   = {}

    # determine unpackable_classes
    if(accelerate_flag):
        unpackable_classes = rel_meta_txt_classes # only exclude `meta` cats
    rel_txt_classes.update(rel_meta_txt_classes)
    
    # load dataset
    dataset = DocDataset(doc_dir=input_dir, meta_only=meta_only, file_type=file_type)
    
    # check if storing figures/tables is feasible
    if(rel_visual_classes and len(dataset) > max_page_to_store_visuals):
        raise DatasetSizeError(len(dataset))
    
    # Create a DataLoader for batching and shuffling
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_yolo,
        shuffle=False,
        collate_fn=custom_collate,
        #num_workers=1,
        #pin_memory=True
    )
    
    # load models
    # - (1.) detection: Yolov5
    detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=detection_weights_path, force_reload=True)
    detect_model.to(device)
    detect_model.eval()

    # - (2.) text classifier for meta data 
    txt_cls_model        = AutoModelForSequenceClassification.from_pretrained(txt_cls_weights_path).to(device)
    tokenizer            = AutoTokenizer.from_pretrained("distilbert-base-uncased", device=device)
    txt_cls_model.eval()
    
    # - (3.) load ViT (i.e. pseudo-OCR) model
    ocr_model     = load_model()
    ocr_processor = load_processor()

    # compile
    ocr_model = torch.compile(ocr_model, fullgraph=True)

    # LaTeX -> Tex Decoder
    LaTex2Text = LatexNodes2Text()

    # track data
    doc_dict = defaultdict(lambda: defaultdict(list))

    # classes
    meta_cls_tensor = torch.tensor(list(rel_meta_txt_classes.values()), device=device, dtype=torch.int)

    # init visual extraction variables
    if(rel_visual_classes):
        i_tab, i_fig, prev_file_id = 0, 0, -1
        all_vis_path_dict = defaultdict(list)
    else:
        vis_path_dict = {}

    # Iterate through the DataLoader
    with torch.no_grad():
        for i,batch in enumerate(data_loader):
            tensors, file_ids, file_paths = batch
            tensors = tensors.to(device)

            # Yolov5 inference (object detecion)
            results = detect_model(tensors)

            # y : dataframe of patch features
            y = pre_processing(results, file_ids, rel_txt_classes.values(), iou_thres=0.001)

            # metadata specific extraction
            pack_patch_tensor, idx_quad, curr_file_ids = get_packed_patch_tensor(tensors, y, rel_txt_classes.values(),
                                                                                 unpackable_class_ids=unpackable_classes.values(),
                                                                                 sep_symbol_flag=False, btm_pad=4, by=['file_id'],
                                                                                 offset=offset, sep_symbol_tensor=None)

            # store visual patches (tables, figures)
            if(rel_visual_classes):
                vis_path_dict, i_tab, i_fig, prev_file_id = store_visuals(tensors, y, rel_visual_classes, file_paths, file_ids,
                                                                          output_dir, i_tab, i_fig, prev_file_id)

            # no use for page images pass this point
            tensors = None

            # ViT: pseudo-OCR inference
            text_results = accelerated_batch_inference(pack_patch_tensor, ocr_model, ocr_processor, batch_size=batch_vit)


            # re-assess meta text categories
            index_quadruplet = assign_text_inferred_meta_classes(txt_cls_model, tokenizer=tokenizer, batch_size=batch_cls, 
                                                                 index_quadruplet=idx_quad, text_results=text_results)

            # assign decoded text to file docs
            doc_dict = update_main_content_dict(doc_dict, text_results, index_quadruplet, curr_file_ids, vis_path_dict)

            # store
            last_batch_flag = (i==len(data_loader)-1)
            store_completed_docs(doc_dict, curr_file_ids, doc_file_paths=dataset.doc_file_paths, store_dir=output_dir, LaTex2Text=LaTex2Text,
                                 store_all_now=last_batch_flag, output_style='text', file_format='jsonl')

    # = = = Done = = =
    
    pass



# entry
if __name__=='__main__':
    main()
