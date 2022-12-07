import os
import requests
import math
import tarfile
import yaml
import sys
from pydub import AudioSegment
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from os.path import join

load_dotenv(find_dotenv())

sys.path.append(join(os.getenv("path.root"), 'src', 'utils'))

from utils import clean_dir  # noqa


EXTERNAL = os.getenv("path.data") + 'external/'
RAW = os.getenv('path.data') + 'raw/'
INTERIM = os.getenv('path.data') + 'interim/'

with open(os.getenv("path.references") + 'GTZAN.yml', 'r') as f:
    DATASET = yaml.load(f)


def download(force=False):
    '''Fetching GTZAN dataset.

        Download GTZAN it is a time-expensive process, therefore:

        If force = False or if we already have a copy of the dataset this
        functions does nothing.

        Otherwise, it will download the GTZAN dataset into the EXTERNAL folder.

        Args:
            force (Boolean): whether to force the download or not.
    '''

    if force or not os.path.isfile(EXTERNAL + 'GTZAN.tar.gz'):
        r = requests.get(DATASET['url'], stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        wrote = 0
        with open(EXTERNAL + 'GTZAN.tar.gz', 'wb') as f:
            for data in tqdm(
                                r.iter_content(block_size),
                                total=math.ceil(total_size//block_size),
                                unit='KB',
                                unit_scale=True
                            ):

                wrote = wrote + len(data)
                f.write(data)

        if total_size != 0 and wrote != total_size:
            print("ERROR, something went wrong")
    else:
        print("\tA local copy of GTZAN was found. Download aborted")


def extract():
    '''Extract the original GTZAN's tar file.

        This function first deletes the RAW/GTZAN folder, then creates a new
        copy extracting EXTERNAL/GTZAN.tar.gz in RAW folder.
    '''
    clean_dir(RAW)
    tf = tarfile.open(EXTERNAL + 'GTZAN.tar.gz')
    tracks = tf.getmembers()

    for track in tqdm(tracks):
        if track.name != 'genres':
            track.name = track.name.replace('genres/', '')
            tf.extract(track, path=RAW)


def encode():
    '''Encodes GTZAN's songs into .wav format.'''

    clean_dir(INTERIM)

    for cat in DATASET['categories']:
        print('\n{0} files: .au -> .wav\n'.format(cat['name'].upper()))
        clean_dir(join(INTERIM, cat['name']))
        tracks = [x for x in os.listdir(join(RAW, cat['name']))]

        for i, track in enumerate(tqdm(tracks)):
            name = track.replace('.au', '.wav')
            song = AudioSegment.from_file(
                os.path.join(RAW, cat['name'], track)
            )
            song.export(join(INTERIM, cat['name'], name), format='wav')

if __name__ == "__main__":

    # DOWNLOAD
    print("\n------------------------------------------------------------")
    print("Downloading GTZAN music genre collection. It may take a while")
    print("Source: {0}".format(DATASET['url']))
    print("------------------------------------------------------------\n")

    download()

    # EXTRACT
    print("\n------------------------------------------------------------")
    print("Extracting GTZAN files")
    print("------------------------------------------------------------\n")

    extract()

    # ENCODE
    print("\n------------------------------------------------------------")
    print("Encoding GTZAN to .wav files")
    print("------------------------------------------------------------\n")

    encode()

    clean_dir(RAW)

    print('')
