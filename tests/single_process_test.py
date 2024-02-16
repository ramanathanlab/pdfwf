import os, yaml
from pathlib import Path
from pdfwf.parsers.oreo import OreoParserConfig, OreoParser

if __name__=='__main__':
    with open('../examples/oreo/oreo_test.yaml') as f:
        config_dict = yaml.safe_load(f)

    # set config
    oreo_config_1 = OreoParserConfig(**config_dict['parser_settings'])

    # init parser
    parser = OreoParser(oreo_config_1)

    # get file paths
    pdf_dir   = Path('/eagle/projects/argonne_tpc/hippekp/small-pdf-set/')
    pdf_files = [pdf_dir / f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    # run
    out = parser.parse(pdf_files)

    # prin
    for i,o in enumerate(out):
        print(f'i={i}')
        print(o)

    print('Finished.')
