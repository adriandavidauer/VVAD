import unittest

import vvadlrs3.utils.downloadUtils as dlUtils
import tempfile
from pathlib import Path


class TestDownloadUtils(unittest.TestCase):
    """
    Test_Download_URL accepts a URL for a video to download and will save that video in
    the output path.
    Output path is defined
    """

    def test_download_url(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Created temporary directory with tmpdirname
            print('created temporary directory', tmpdirname)
            temp_dir = Path(tmpdirname)
            file_dir = temp_dir.joinpath("test.mp4")
            dlUtils.download_url('https://www.youtube.com/watch?v=00j9bKdiOjk',
                                 file_dir)

            self.assertEqual(file_dir.exists(), True)
            # ~ ... /tmp/tmp81iox6s2/test.mp4 True


if __name__ == '__main__':
    unittest.main(warnings='ignore')
