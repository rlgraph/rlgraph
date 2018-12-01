FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && apt-get -y install git libsm6 libxrender-dev libxext6 cmake libz-dev
RUN curl https://bootstrap.pypa.io/get-pip.py | python
RUN python -m pip install -U pip
RUN python -m pip install -U setuptools
RUN python -m pip install -U numpy
RUN python -m pip install opencv-python psutil pytest-xdist
RUN python -m pip install pytest-xdist
RUN python -m pip install pyyaml six requests lz4
RUN python -m pip install gym atari_py box2d-py
RUN python -m pip install ray

# In the future, add further extra packages here, like horovod, pytorch, or ray
# TODO: Use our `extras_require` in setup.py, instead of specifying individual packages.
#RUN cd /rlgraph && python -m pip install gym atari_py
#RUN cd /rlgraph && python -m pip install ray

CMD ["bash"]
