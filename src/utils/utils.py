import sys
import os
import shutil
import numpy as np
import librosa as lbr
import yaml

with open(os.getenv("path.references") + "GTZAN.yml", "r") as f:
    DATASET = yaml.load(f)

def read_args(rs=42, use='MEL', dim='1D', batch=1):
    ARGS = {
        'rs': rs,
        'dim': dim,
        'use': use,
        'batch': batch
    }

    args = sys.argv[1:]
    args = map(lambda x: x.split("="), args)
    args = filter(lambda x: x[0] in ARGS.keys(), args)

    for arg in args:
        if arg[0] in ARGS.keys():
            ARGS[arg[0]] = arg[1] if arg[1] != "" else ARGS[arg[0]]
        if str(ARGS[arg[0]]).isnumeric():
            ARGS[arg[0]] = int(ARGS[arg[0]])

    return ARGS


def clean_dir(path):
    ''' If path exists, it deletes it and then creates a new copy

        Arguments:
            - path(str): directory's path
    '''
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
