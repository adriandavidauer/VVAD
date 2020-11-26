from setuptools import setup, find_packages
# import pathlib
import os

# The directory containing this file
# HERE = pathlib.Path(__file__).parent
HERE = os.path.dirname(__file__)

# The text of the README file
readmePath = os.path.join(HERE , "README.md")
README = ""
with open(readmePath) as readmeFile:
    README = readmeFile.read()


if 'VERSION' in os.environ:
    VERSION = os.environ['VERSION']
else:
    VERSION = '0.0.0'

# VERSION = (HERE / "VERSION").read_text()
# try:
#     VERSION += '.{}'.format(os.environ["CI_PIPELINE_IID"])
# except:
#     print('LOCAL BUILD')



setup(name='vvadlrs3',
    version=VERSION,
    description="Provides tools to handle the VVAD-LRS3 dataset.",
    # long_description=README,
    # long_description_content_type="text/markdown",
    url="https://www.kaggle.com/adrianlubitz/vvadlrs3",
    author="Adrian Lubitz",
    author_email="adrianlubitz@gmail.com",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3.7",
    ],
    packages = find_packages(exclude='code'),
    # py_modules=['pretrained_models.py'],
    install_requires=[
        "keras==2.4.3", 
        "tensorflow==2.3.1"
    ],
    package_data={'': ['*.h5']},
    include_package_data=True,
    python_requires='>=3.7',
)