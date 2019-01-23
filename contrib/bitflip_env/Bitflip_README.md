#BitFlip Environment


BitFlip is a custom environment that was used in HER paper https://arxiv.org/abs/1707.01495

BitFlip is an environment with the state space S = {0, 1}^n
and action space {A = 0 <= i <= n - 1} where n >= 0 and i refers to flipping the *i*th bit in the state space.
When i = n the environment process one step without flipping any bits.
Every environment has a goal state *g*, where *g* is sampled from the possible states. 
The reward function is defined such that policy receive 0 if the s = g, -1 otherwise.