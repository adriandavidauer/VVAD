import os.path
import shutil
import urllib.request
import zipfile
from tqdm import tqdm

"""
Initializing the URLs and file names for the WildVVAD video files to download
"""
speaking_videos_url = \
    "http://perception.inrialpes.fr/Free_Access_Data/WVVAD/speaking_videos.zip"
speaking_videos_file = 'speaking_videos.zip'
speaking_videos_folder = 'speaking_videos'

silent_videos_url = \
    "http://perception.inrialpes.fr/Free_Access_Data/WVVAD/silent_videos.zip"

silent_videos_file = 'silent_videos.zip'
silent_videos_folder = 'silent_videos'


def __init__(self):
    pass


def download_and_save_speaking_videos() -> None:
    """
    This method will download, save, and extract all speaking_videos videos from
    the WildVVAD data set.
    """

    print("Download speaking_videos videos.")
    download_and_save_videos(speaking_videos_url,
                             speaking_videos_folder,
                             speaking_videos_file)


def download_and_save_silent_videos() -> None:
    """
    This method will download, save, and extract all silent_videos videos from
    the WildVVAD data set.
    """
    print("Download silent_videos videos.")
    download_and_save_videos(silent_videos_url,
                             silent_videos_folder,
                             silent_videos_file)


def download_and_save_videos(url: str, folder_name: str, file_name: str) -> None:
    """
    General method to download, save, and extract zip files from URLs


    Args:
        url (str): URL to zip file to download
        folder_name (str): Folder to save the file into
        file_name (str): Name to save the file as
    """

    file_exists = False
    while not file_exists:
        try:
            with urllib.request.urlopen(url) as response, open(
                    os.path.join(folder_name, file_name), 'wb') as out_file:  #
                file_exists = True
                shutil.copyfileobj(response, out_file)
                with zipfile.ZipFile(
                        os.path.join(folder_name, file_name)) as zf:
                    zf.extractall()
        except FileNotFoundError:
            os.mkdir(folder_name)


class DownloadProgressBar(tqdm):
    """
    Download progress bar used during zip file download. Indicating progress.
    """

    def __init__(self):
        self.total = None

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if __name__ == "__main__":
    # Testing:
    download_and_save_speaking_videos()
