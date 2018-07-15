FROM ubuntu:xenial

ARG py=3.5

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends tzdata shared-mime-info sudo vim git python$py python${py}-dev python3-pip python3-setuptools
RUN apt-get install -y --no-install-recommends cmake clang-5.0 clang-3.8 git build-essential ca-certificates pkg-config bash-completion
RUN pip3 install numpy pydevd tensorflow cached-property gym gym[Atari] scipy pyyaml ray opencv-python pyyaml pytest six requests lz4 pyarrow

CMD ["bash"]
