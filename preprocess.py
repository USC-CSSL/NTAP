"""
***Input pipeline for DATAM***

Params:
    --input: Source file (csv, tsv, pkl) with header and valid index
    --output: Destination directory
    --params: Path to parameter JSON file
"""
import pandas as pd
import argparse, os
from process.processor import Preprocessor
import json

# TAGME_QCODE = os.environ["TAGME"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to source file")
    parser.add_argument("--jobs", help="Options: clean pos ner deparse tagme", 
                        nargs='+')
    parser.add_argument("--output", help="Path to destination directory")
    args = parser.parse_args()

    with open('params.json') as f:
        params = json.load(f)

    processor = Preprocessor(args.output, params)
    try:
        processor.load(args.input)
    except Exception as e:
        print(e)
        print("Could not load data from {}".format(args.input))
        exit(1)
   
    for job in args.jobs:
        print("Processing job: {}".format(job))
        if job == 'clean':
            processor.clean(params["processing"]["clean"], remove=True)
        if job == 'ner':
            processor.ner()
        if job == 'pos':
            processor.pos()
        if job == 'depparse':
            processor.depparse()
        if job == 'tagme':
            processor.tagme()
    ftype = ".pkl"
    processor.write(ftype)

