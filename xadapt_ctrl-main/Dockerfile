# Use the official ROS Noetic image as base
FROM ubuntu:22.04

# Set noninteractive mode to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libffi-dev \
    libbz2-dev \
    libssl-dev \
    libncurses5-dev \
    libsqlite3-dev \
    libreadline-dev \
    tk-dev \
    zlib1g-dev \
    xz-utils \
    python3-tk \
    x11-apps \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y python3-tk

# Set project path inside the container
WORKDIR /root/xadapt_ctrl

# Copy the entire project into the container
COPY . .

# Ensure setup.bash exists before proceeding
RUN test -f setup.bash || (echo "please launch from the xadapt_ctrl folder!" && exit 1)

# Update submodules
RUN git submodule update --init --recursive

# Install Python dependencies (py3dmath)
WORKDIR /root/xadapt_ctrl/py3dmath
RUN pip install -e .

# Install required Python packages using pip
WORKDIR /root/xadapt_ctrl
RUN pip install --no-cache-dir \
    coloredlogs==15.0.1 \
    contourpy \
    cycler==0.12.1 \
    flatbuffers==24.3.25 \
    fonttools==4.53.1 \
    humanfriendly==10.0 \
    kiwisolver==1.4.5 \
    matplotlib \
    mpmath==1.3.0 \
    numpy\
    onnx==1.16.2 \
    onnxruntime \
    packaging==24.1 \
    pandas \
    pillow==10.4.0 \
    protobuf==5.28.0 \
    pyparsing==3.1.4 \
    python-dateutil==2.9.0.post0 \
    pytz==2024.1 \
    scipy \
    six==1.16.0 \
    sympy==1.13.2 \
    tzdata==2024.1

# Run the simulation
CMD ["bash", "-c", "echo 'Have a safe flight!'"]
