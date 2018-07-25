##### How to run openAI gym Atari envs in Win10 Docker containers and render the game.
1) Start the XMing Server without(!) the access control option.
2) docker run -it -e DISPLAY=192.168.2.107:0 rlgraph/rlgraph:py3.6
