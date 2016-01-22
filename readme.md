

# Assumption Mining for Monotone Systems

The code used to generate the figures for the paper _Directed Specifications and Assumption Mining for Monotone Dynamical Systems_, to appear at HSCC 2016.

**Requirements**
- python 2.7
- ipython
- z3 theorem prover
- line-profiler

**Installation Instructions**
- Install the [Anaconda](https://anaconda.org/) python 2.7 distribution, which comes with the `conda` package manager and ipython. 
- Timing profiler magic commands. Installation instructions found [here](http://pynash.org/2013/03/06/timing-and-profiling/)
- Install the [z3 theorem prover](https://github.com/Z3Prover/z3/wiki) with python API by running `conda install -c https://conda.anaconda.org/asmeurer z3`
  If z3 still does not install you can try calling `pip install angr-z3`

##TODO
- [ ] Clean up code in and provide additional documentation for `AGmining`, `trafficCTM`, and `stl` packages
- [ ] Write unit tests