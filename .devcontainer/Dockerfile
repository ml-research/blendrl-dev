# Select the base image
FROM nvcr.io/nvidia/pytorch:24.03-py3
# FROM pytorch/pytorch
# FROM ubuntu/python

RUN apt-get update

RUN apt-get install -y git

RUN pip install --upgrade pip

RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get install ffmpeg libsm6 libxext6  -y