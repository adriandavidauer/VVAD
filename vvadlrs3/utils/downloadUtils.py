''' Utils for downloads '''

# System imports
from pathlib import Path
import urllib.request
import bz2
import errno
import os


# 3rd Party imports
from tqdm import tqdm

# local imports

# end file header

__author__ = 'Adrian Lubitz'


# https://stackoverflow.com/a/53877507


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)
