import sys
import os
from dotenv import find_dotenv, load_dotenv
import random

load_dotenv(find_dotenv())

sys.path.append(os.path.join(os.getenv("path.root"), 'src', 'features'))
sys.path.append(os.path.join(os.getenv("path.root"), 'src', 'utils'))

from utils import read_args  # noqa
from Features import Feature  # noqa

if __name__ == "__main__":

    ARGS = read_args(rs=42, dim=1, use='MEL')

    feat = Feature(rs=ARGS['rs'], use=ARGS['use'], dim=ARGS['dim'])

    # CREATE FEATURES
    print("\n------------------------------------------------------------")
    print("\tCreating a TRAIN/TEST split (RS: {0})".format(ARGS['rs']))
    print("------------------------------------------------------------\n")

    feat.read().save()

    print("\n")
