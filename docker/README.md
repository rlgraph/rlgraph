# RLgraph Dockerfiles: HowTo.

#### 1) Which Docker files are available via dockerhub?
We currently have the following docker files ready for download from dockerhub.
To download our latest one-fits-all container, do:

`docker pull rlgraph/rlgraph:latest`

##### a) `rlgraph/rlgraph:latest`
This is our general purpose, one-fits-all container. It contains all the libraries
and other dependencies needed to run any RLgraph experiment under python3.
This includes (only major packages listed):
- python3
- tensorflow (CPU version, currently 1.13.1)
- pytorch (version )
- ray (needed for certain Agents, e.g. ApeX)
- openAI gym environments (including Atari support)
- deepmind Lab environment


#### 2) How to render environments such as openAI gym Atari or deepmind Lab from inside a Win10 Docker container.
A common problem for Windows users is the incompatibility of their OS with many
libraries used in ML/RL research (e.g. ray, gym\[atari\], etc..). In order to still be able
to run such libraries, Win users can pull our RLgraph-ready docker containers from dockerhub
(see (1)) and even get a Windows display of the rendered content running inside the container.

To do so, follow these simple steps:
1) Install and start the XMing Server on your Windows host system, without(!) the "access control" option set.
2) In a command shell, execute:
```
$ipconfig
```

This will tell you the IP address of your host Windows system. It is usually
a non-public IP address, starting with `192.168....`.

```
$docker pull rlgraph/rlgraph:latest
$docker run -it -e DISPLAY=[IP address from step above]:0 rlgraph/rlgraph:latest
```

