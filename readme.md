

# Assumption Mining for Monotone Systems

The code used to generate the figures for the paper _Directed Specifications and Assumption Mining for Monotone Dynamical Systems_, to appear at HSCC 2016.

An ipython notebook showcasing the results from the paper is found at []() and 

**Runtime Requirements**
In addition to python 2.7 and [ipython](http://ipython.org/), you'll need the following:
- Timing profiler magic commands. Installation instructions found [here](http://pynash.org/2013/03/06/timing-and-profiling/)
- The [z3 theorem prover](https://github.com/Z3Prover/z3/wiki) with python API. On Mac OSX with [homebrew](http://brew.sh/), you can install it by running `brew install z3` in the terminal.

##TODO
- [ ] Clean up code in and provide documentation for `AGmining`, `trafficCTM`, and `stl` packages
- [ ] Write unit tests
- [ ] 