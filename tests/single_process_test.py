from __future__ import annotations

import sys
from pathlib import Path

import yaml

from pdfwf.parsers.oreo import OreoParser
from pdfwf.parsers.oreo import OreoParserConfig

# TODO: Make this an official test

if __name__ == '__main__':
    with open('../examples/oreo/oreo_test.yaml') as f:
        config_dict = yaml.safe_load(f)

    # set config
    oreo_config_1 = OreoParserConfig(**config_dict['parser_settings'])

    # init parser
    parser = OreoParser(oreo_config_1)

    # get file paths
    pdf_dir = Path('/eagle/projects/argonne_tpc/hippekp/small-pdf-set/')
    pdf_files = [str(f) for f in pdf_dir.glob('**/*.pdf')]

    # run
    out = parser.parse(pdf_files)

    if out is None:
        print('Parsing failed.')
        sys.exit(1)

    # prin
    for i, o in enumerate(out):
        print(f'i={i}')
        print(o)

    print('Finished.')
