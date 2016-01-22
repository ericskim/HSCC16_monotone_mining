

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

**Run Instructions**
The best way to run the examples is by executing the two ipython notebooks, `freeway_example.ipynb` and `integrator_example.ipynb`. You can also view the results of executing the [freeway_example](http://nbviewer.jupyter.org/url/www.eecs.berkeley.edu/~eskim/jupyter_notebooks/freeway_example.ipynb#) notebook and [integrator](http://nbviewer.jupyter.org/url/www.eecs.berkeley.edu/~eskim/jupyter_notebooks/integrator_example.ipynb#) example notebook without running them.

After following all of the installation instructions above, run `ipython notebook` in this directory.

##TODO
- [ ] Clean up code in and provide additional documentation for `AGmining`, `trafficCTM`, and `stl` packages
- [ ] Write unit tests