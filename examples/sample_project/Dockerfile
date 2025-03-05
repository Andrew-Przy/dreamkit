FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
    vim libgl1-mesa-glx libosmesa6 \
    wget unrar cmake g++ libgl1-mesa-dev \
    libx11-6 openjdk-8-jdk x11-xserver-utils xvfb \
    && apt-get clean

# Optionally, copy your application files here
COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

# Command to keep the container running indefinitely
CMD ["tail", "-f", "/dev/null"]
