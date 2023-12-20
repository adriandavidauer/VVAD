''' Utils for downloads '''

# System imports
import urllib.request

# 3rd Party imports
from tqdm import tqdm

# local imports

# end file header

__author__ = 'Adrian Auer'


# https://stackoverflow.com/a/53877507


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """ Download a video file from a given URL.
        In this case, it is about youtube videos which will be saved as
        *.3gpp files.

    Args:
        url (str): URL to video to download
        output_path (str): Path to directory in which the downloaded file should be
            saved
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)
