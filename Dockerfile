# Set python version
FROM python:3.7
LABEL authors="kzanaty"

RUN apt-get update
RUN apt-get install git ffmpeg -y
RUN apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev -y

RUN python --version

RUN python -m pip install numpy
RUN python -m pip install vvadlrs3
# anything else to install?

# set work directory
WORKDIR /home

# create folder for dataset output and tmp files
RUN mkdir dataset output tmp

ENTRYPOINT ["echo", "'test'"]