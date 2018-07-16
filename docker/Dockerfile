FROM ubuntu:artful

ARG py=3.6

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y python$py
RUN apt-get install -y --no-install-recommends tzdata shared-mime-info sudo vim git python${py}-dev python3-pip python3-setuptools
RUN apt-get install -y --no-install-recommends cmake clang-5.0 clang-3.8 git build-essential ca-certificates pkg-config bash-completion
RUN apt-get install -y libopencv-dev

# Needed for rendering openAI gym[atari]-Envs in Win Docker container using xMing.
RUN apt-get install -y freeglut3-dev

RUN pip3 install gym[atari]
RUN pip3 install numpy pydevd tensorflow cached-property gym scipy pyyaml
RUN pip3 install ray
RUN pip3 install opencv-python pyyaml pytest six requests lz4 pyarrow
RUN pip3 install wheel

CMD ["bash"]
