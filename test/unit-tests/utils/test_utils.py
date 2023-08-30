import unittest

from test_downloadUtils import *
from test_imageUtils import *
from test_kerasUtils import *
from test_multiprocessingUtils import *
from test_plotUtils import *
from test_timeUtils import *
from test_videoUtils import *

suite_download = unittest.TestLoader().loadTestsFromTestCase(TestDownloadUtils)
suite_image = unittest.TestLoader().loadTestsFromTestCase(TestImageUtils)
all_tests = unittest.TestSuite([suite_download, suite_image])
