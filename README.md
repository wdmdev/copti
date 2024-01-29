# COpti
My journey into constrained optimization.

Based on my learnings in the [DTU](https://www.dtu.dk/english) course [Constrained Optimization](https://kurser.dtu.dk/course/02612).

## Purpose
I'm creating this python package mostly for my own use at the exam and to motivate myself to implement my learnings in practice.

If the package seems of interest to anyone else fell free to use it (the license is MIT).

## Installation
The project depends on [`cyipopt`](https://pypi.org/project/cyipopt/) which uses the [`IPOPT`](https://coin-or.github.io/Ipopt/) software package.

You can install IPOPT directly using their [guide](https://coin-or.github.io/Ipopt/INSTALL.html).

However, I would recommend using `conda` if possible since it's a lot simpler:
```
conda install -c conda-forge cyipopt
```
