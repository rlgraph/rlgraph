# Image for rlgraph/rlgraph:full
# Contains all RLgraph supported envs.

FROM ubuntu:cosmic

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install -y python3.6
# Create proper softlink to py3
WORKDIR /usr/bin
RUN rm -rf python && ln -s python3 python

# Install all necessary packages to run any RLgraph experiment in any RLgraph supported environment.
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata shared-mime-info sudo vim git python3.6-dev python3-pip \
    python3-setuptools libopencv-dev cmake clang-6.0 git build-essential ca-certificates \
    pkg-config bash-completion lua5.1 liblua5.1-0-dev libffi-dev gettext freeglut3-dev libsdl2-dev libjpeg-dev \
    libosmesa6-dev python3-pil python3-numpy jq manpages-pl manpages-fr-extra
    #realpath  # clang-3.8

# Vizdoom stuff
RUN apt-get install -y --no-install-recommends nasm tar libbz2-dev libgtk2.0-dev libfluidsynth-dev libgme-dev \
    libopenal-dev timidity libwildmidi-dev libboost-all-dev liblua5.1-0-dev julia

# Needed for rendering openAI gym[atari]-Envs in Win Docker container using xMing.
RUN apt-get install -y freeglut3-dev

# Needed for Deepmind Lab.
RUN apt-get install -y curl zip unzip g++ zlib1g-dev openjdk-8-jdk
RUN apt-get install -y gnupg
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list; curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
RUN sudo apt-get update -y && sudo apt-get install -y bazel
WORKDIR /root/
RUN git clone https://github.com/deepmind/lab.git
WORKDIR lab/
RUN touch WORKSPACE

# Upgrade pip before installing packages with it.
RUN pip3 install --upgrade pip
# Install necessary pip packages.
RUN pip3 install numpy pydevd tensorflow==1.13.1 tensorflow_probability cached-property gym scipy pyyaml gym[atari] ray opencv-python pytest six \
    requests lz4 pyarrow wheel atari_py torch torchvision pygame
RUN pip3 install graphviz

# Write a correct bazel build config for python so that bazel finds the python and numpy headers.
RUN echo 'cc_library(name = "python",hdrs = glob(["include/python3.6/**/*.h"]),' \
    'includes = ["include/python3.6", "include/python3.6/numpy"], visibility = ["//visibility:public"],)' > \
    python.BUILD

# Bazel-build and install deepmind Lab.
RUN bazel build -c opt python/pip_package:build_pip_package

# TODO: Why do we need to do this again here?: Create proper softlink to py3
WORKDIR /usr/bin
RUN rm -rf python && ln -s python3 python
WORKDIR /root/lab/

RUN ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
RUN pip3 install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py3-none-any.whl --force-reinstall

# Compile the batcher.cc file into batcher.so
#RUN TF_INC="$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')" && \
#    TF_LIB="$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')" && \
#    g++-4.8 -std=c++11 -shared batcher.cc -o batcher.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -ltensorflow_framework

CMD ["bash"]
