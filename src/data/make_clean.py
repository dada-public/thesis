import sys
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

sys.path.append(os.path.join(os.getenv("path.root"), 'src', 'utils'))

from utils import clean_dir  # noqa


if __name__ == "__main__":
    clean_dir(os.getenv("path.models"))
    clean_dir(os.getenv("path.data.interim"))
    clean_dir(os.getenv("path.data.raw"))
    clean_dir(os.path.join(os.getenv("path.data.processed"), 'MEL'))
    clean_dir(os.path.join(os.getenv("path.data.processed"), 'CHROMA'))
    clean_dir(os.path.join(os.getenv("path.data.processed"), 'SPEC'))
