import sys
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

sys.path.append(os.path.join(os.getenv("path.root"), 'src', 'utils'))
sys.path.append(os.path.join(os.getenv("path.root"), 'src', 'models'))

from utils import read_args  # noqa
from Model import Model1D, Model2D  # noqa

if __name__ == "__main__":

    ARGS = read_args(rs=42, use='MEL', dim=1, batch=1)

    print("\n------------------------------------------------------------")
    print("\tTraining a {0}D model using {1}".format(ARGS['dim'], ARGS['use']))
    print("------------------------------------------------------------\n")

    if ARGS['dim'] == 1:
        model = Model1D(rs=ARGS['rs'], use=ARGS['use'])
    else:
        model = Model2D(rs=ARGS['rs'], use=ARGS['use'])

    model.plot().train().save(os.getenv("path.models"))
