# ---------- STAGE 1: Base ----------
# Select the base image
FROM nvcr.io/nvidia/pytorch:24.03-py3 AS base
# FROM pytorch/pytorch
# FROM ubuntu/python

RUN apt-get update

RUN apt-get install -y git

RUN pip install --upgrade pip

RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

# Install requirements
COPY requirements.txt .

RUN pip install -r requirements.txt

# Install NSFR
COPY nsfr nsfr
RUN pip install -e nsfr

# Install NUDGE
COPY nudge nudge
RUN pip install -e nudge

# Install NEUMANN
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

COPY neumann neumann
RUN pip install -e neumann

# Install difflogic
COPY third_party/difflogic third_party/difflogic
RUN pip install -e third_party/difflogic

# Install hackatari
COPY third_party/hackatari third_party/hackatari
RUN pip install -e third_party/hackatari
