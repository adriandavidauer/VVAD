'''
Installer for the vvad-lrs3-lib
'''

# System imports
import os
from setuptools import setup
# 3rd party imports

# local imports

# end file header
__author__      = 'Adrian Lubitz'




# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if 'RELEASE_VERSION' in os.environ:
    VERSION = os.environ['RELEASE_VERSION']
else:
    VERSION = '0.0.0'

setup(
    name = "vvadlrs3",
    version = VERSION, # TODO: version should be set from tag or whatever
    author = "Adrian Lubitz",
    author_email = "adrianlubitz@gmail.com",
    description = ("Library to provide models trained on the VVAD-LRS3 Dataset. The library also contains preprocessing pipelines."),
    license = "LGPLv2. Note that the license for the iBUG 300-W dataset which was used for face and lip features excludes commercial use.",
    keywords = "VVAD LRS3 AI Social robotics",
    url = "https://github.com/adrianlubitz/VVAD",
    packages=['vvadlrs3'],
    package_data={
        # Include all h5 files in the vvadlrs3 package
        "vvadlrs3": ["*.h5"],
    },
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
    ],
    install_requires=[
          'numpy', # TODO: Which are the needed dependencies
          'opencv-contrib-python',
          'dlib', # TODO: This guy is problematic because it needs cmake...That needs to be communicated somewhere
          'matplotlib',
          'keras==2.4.3', 
          'tensorflow==2.3.1'
      ],
    python_requires='>=3.7',
)