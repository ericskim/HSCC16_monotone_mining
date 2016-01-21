import numpy as np 
import z3
import types


class integrator(object):
    def __init__ (self, a, x0 = None):
        if a < 0:
            raise ValueError('a must be nonnegative')
        self.a = a
        self.x0 = x0
        self.state_dim = 1
        self.input_dim = 1
        self.output_dim = 1
        self.state_order = np.ones([self.state_dim])
        self.input_order = np.ones([self.input_dim])
        self.output_order = np.ones([self.output_dim])

    def simulate(self, u, x0):
        
        x = np.zeros([len(u)+1, 1])
        x[0] = x0

        for t in range(len(u)):
            x[t + 1] = max(0, self.a * x[t] + u[t])

        return x

    def output_map(self, x):
        """
        Takes state trajectories and applies an output function 
        """
        #raise NotImplementedError(" ")
        return x
        

    def z3_dynamics_constraints(self, T):
        """
        Get dynamical system constraints.

        output
        -----
        constraints: z3 dynamics constraints
        x: numpy array of z3 state variables
        u: numpy array of z3 input variables

        """

        def z3max(assignment, array):
            assert (isinstance(array,types.TupleType) or isinstance(array,types.ListType))

            constraints = [assignment >= array[i] for i in range(len(array))]
            equal_to_constraint = z3.Or([assignment == array[i] for i in range(len(array))])
            constraints.append(equal_to_constraint)

            return constraints

        def z3min(assignment, array):
            assert (isinstance(array,types.TupleType) or isinstance(array,types.ListType))

            constraints = [assignment <= array[i] for i in range(len(array))]
            equal_to_constraint = z3.Or([assignment == array[i] for i in range(len(array))])
            constraints.append(equal_to_constraint)

            return constraints

        x = np.array([[z3.Real("x%i" % i)] for i in range(T+1)])
        u = np.array([[z3.Real("u%i" % i)] for i in range(T)])

        assert(self.state_dim == x.shape[1]), "z3 state variable dimension inconsistent"
        assert(self.input_dim == u.shape[1]), "z3 input variable dimension inconsistent"

        constraints = [x[i,0] >= 0 for i in range(T+1)]

        for i in range(T):
            dynamics_constraint = z3max(x[i+1,0], (0, self.a * x[i,0] + u[i,0]))
            constraints.extend(dynamics_constraint)

        if self.x0 is not None:
            for i in range(self.state_dim):
                initial_constraint = [x[0,i] == self.x0[i]]
        constraints.extend(initial_constraint)

        return constraints, x, u

    def z3_output_constraints(self, x):
        """
        Takes z3 state variables and imposes output constraints

        """

        constraints = []
        T = x.shape[0]
        state_dim = x.shape[1]
        output_dim = self.output_dim

        y = np.array([[z3.Real("y%i_%i" % (i,t)) for i in range(self.output_dim)] for t in range(T)])

        constraints = np.array([[y[t,i] == self.output_map(x[t])[i] for i in range(output_dim)] for t in range(T)])
        constraints = (constraints.flatten()).tolist()

        return constraints, y 


