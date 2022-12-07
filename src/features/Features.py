import os
import sys
import numpy as np
import numpy
import yaml
import librosa as lbr
import random

from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from os.path import join
from shutil import copyfile

load_dotenv(find_dotenv())

sys.path.append(join(os.getenv("path.root"), 'src', 'utils'))

from utils import clean_dir  # noqa

with open(os.getenv("path.references") + "GTZAN.yml", "r") as f:
    DATASET = yaml.load(f)

INTERIM = os.getenv("path.data.interim")


def feats(use='MEL'):
    ''' Features available: CHROMAGRAMS, MELSPECTOGRAM & QT-transforms '''
    fmin = lbr.midi_to_hz(36)
    sr = DATASET['sample_rate']
    feat = {
        'MEL': lambda x: lbr.feature.melspectrogram(x, sr=sr),
        'QT': lambda x: lbr.cqt(x, sr=sr, fmin=fmin),
        'CHROMA': lambda x: lbr.feature.chroma_cens(x, sr=sr)
    }
    return feat[use]


def trim(sig, cutoff=660000):
    ''' Inspite each song is 30 seconds long, not all of them have the
        same length
    '''
    return sig[:cutoff]


def chunks(sig, window=0.1, overlap=0.5):
    ''' We cut each song in chunks of window*30s length, overlapped
        by half of its length
    '''
    X = []
    sig_shape = sig.shape[0]
    chunk = int(sig_shape*window)
    offset = int(chunk*(1.-overlap))

    for i in range(0, sig_shape - chunk + offset, offset):
        X.append(sig[i:i + chunk])

    return np.array(X)


class Feature:

    def __init__(self, rs=42, use='MEL', dim=1, src=INTERIM):
        self.use = use
        self.src = src
        self.train = []
        self.test = []
        self.dim = dim

        random.seed(rs)

    def read(self):
        ''' It reads the original dataset and splits it into a
            TRAIN (70%) / TEST (30%)
        '''
        for cat in DATASET['categories']:
            songs = os.listdir(join(self.src, cat['name']))
            songs = random.sample(songs, len(songs))
            split = int(len(songs)*.7)
            for song in songs[:split]:
                self.train.append({
                    'name': str(cat['klass']) + '_' + song,
                    'klass': cat['klass'],
                    'src': join(self.src, cat['name'], song)
                })
            for song in songs[split:len(songs)]:
                self.test.append({
                    'name': str(cat['klass']) + '_' + song,
                    'klass': cat['klass'],
                    'src': join(self.src, cat['name'], song)
                })

        return self

    def save(self, dest=os.getenv("path.data.processed")):
        ''' It saves the results in .npy format'''
        TRAIN_PATH = join(dest, self.use, 'train')
        TEST_PATH = join(dest, self.use, 'test')

        print("\nCreating TEST dataset ...\n")
        clean_dir(TEST_PATH)
        for index, song in enumerate(self.test):
            copyfile(song['src'], join(TEST_PATH, song['name']))

        print("\nCreating TRAIN dataset ...\n")
        clean_dir(TRAIN_PATH)
        X = []
        Y = []
        for song in tqdm(self.train):
            data, labels = self.process(song['src'], song['klass'])
            X.extend(data)
            Y.extend(labels)

        np.save(join(TRAIN_PATH, 'x.npy'), np.array(X))
        np.save(join(TRAIN_PATH, 'y.npy'), np.array(Y))

        return self

    def process(self, src, klass):
        ''' It applies the data augmentation and computes the selected
            features
        '''
        signal, sr = lbr.load(src)
        signal = trim(signal)
        excerpts = []

        for excerpt in chunks(signal):
            if self.dim == 1:
                excerpts.append(feats(use=self.use)(excerpt))
            else:
                excerpts.append(feats(use=self.use)(excerpt)[:, :, np.newaxis])
        excerpts = np.array(excerpts)

        labels = [np.eye(10, k=klass, dtype=int)[0]] * len(excerpts)

        return excerpts, labels
