import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from os.path import join, isfile
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import recall_score, precision_score


load_dotenv(find_dotenv())

sys.path.append(os.path.join(os.getenv("path.root"), 'src', 'utils'))
sys.path.append(os.path.join(os.getenv("path.root"), 'src', 'models'))

from utils import read_args  # noqa
from Model import Model1D, Model2D  # noqa

if __name__ == "__main__":

    ARGS = read_args(rs=42, use='MEL', dim=1)

    if ARGS['dim'] == 1:
        model = Model1D(rs=ARGS['rs'], use=ARGS['use'])
    else:
        model = Model2D(rs=ARGS['rs'], use=ARGS['use'])

    model.load("{0}_{1}D".format(ARGS['use'], str(ARGS['dim'])))

    TEST_DIR = join(os.getenv("path.data.processed"), ARGS['use'], 'test')
    songs = [join(TEST_DIR, x) for x in os.listdir(TEST_DIR)]

    print("\n------------------------------------------------------------")
    print("Evaluating a {0}D model using {1}".format(ARGS['dim'], ARGS['use']))
    print("------------------------------------------------------------\n")

    klasses = []
    predictions = []

    for song in tqdm(songs):
        klasses.append(int(os.path.basename(song).split('_')[0]))
        predictions.append(model.eval(song))

    print("\n------------------------------------------------------------")
    print("Results:")
    print("------------------------------------------------------------\n")

    cm = confusion_matrix(klasses, predictions)

    RESULTS = join(
        os.getenv("path.results"),
        ARGS['use'],
        str(ARGS['dim']) + 'D'
    )

    df = pd.DataFrame({
        'K0': cm.T[0], 'K1': cm.T[1], 'K2': cm.T[2], 'K3': cm.T[3],
        'K4': cm.T[4], 'K5': cm.T[5], 'K6': cm.T[6], 'K7': cm.T[7],
        'K8': cm.T[8], 'K9': cm.T[9],
        'pre': precision_score(klasses, predictions, average=None),
        'rec': recall_score(klasses, predictions, average=None),
        'acc': [accuracy_score(klasses, predictions)] * 10
    })

    print("\nConfusion matrix & per-class metrics:\n")
    print(df)
    print("\nACC: {0:.2f}, PREC: {1:.2f}, REC: {2:.2f}\n".format(
            df['acc'].mean(), df['pre'].mean(), df['rec'].mean()
        )
    )

    if not os.path.isdir(RESULTS):
        os.makedirs(RESULTS)
    df.to_csv(join(RESULTS, str(ARGS['rs']) + '.csv'), index=False)
