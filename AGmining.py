#!/usr/bin/env python

import time
import pdb
import types
import warnings

import numpy as np

import z3

def not_in_rectangle(z3_x, point, rect_direction, epsilon = 0, order = None):
    """
    Create a z3 disjunction constraint from a point with bloating epsilon.

    Args:
        z3_x (list of z3 variable): 
        point (1D numpy array): 
    """
    ndim = len(point)
    rect_disjunction = [None] * ndim

    if order is None:
        #order = np.ones([ndim]) # Default coordinate-wise order
        order = [1.0] * ndim 
    else:
        order = order.tolist()

    if rect_direction is 'lower':
        for dim in range(ndim):
            o = order[dim] 
            rect_disjunction[dim] = (o*z3_x[dim] >= o*point[dim] + epsilon) # best
    elif rect_direction is 'upper':
        for dim in range(ndim):
            o = order[dim]
            rect_disjunction[dim] = (o*z3_x[dim] <= o*point[dim] - epsilon)     
    else:
        raise ValueError("rect_direction must be 'lower' or 'upper' ")
    
    return z3.simplify(z3.Or(rect_disjunction))

def in_rectangle(z3_x, point, rect_direction, epsilon = 0, order = None):
    """
    Create a z3 disjunction constraint from a point with bloating epsilon.
    """
    rect_conjunction = []
    ndim = len(point)

    if order is None:
        order = np.ones([ndim]) # Default coordinate-wise order
    else:
        order = order.tolist()

    if rect_direction is 'lower':
        for dim in range(ndim):
            o = order[dim]
            rect_conjunction.append(o*z3_x[dim] <= o*(point[dim] - o*epsilon)) # big slowdown here
    elif rect_direction is 'upper':
        for dim in range(ndim):
            o = order[dim]
            rect_conjunction.append(o*z3_x[dim] >= o*(point[dim] + o*epsilon)) # big slowdown here
    else:
        raise ValueError("rect_direction must be 'lower' or 'upper' ")

    return z3.And(rect_conjunction)

def z3_to_float(x):
    """
    Converts z3 variable to a floating point variable
    """
    return float(x.numerator_as_long())/float(x.denominator_as_long())

class assumption_miner(object):
    """
    Assumption Miner Class. 

    Given a specification, determines the 

    Args:
        system: System class that is used to generate simulated trajectories
        T (int): Disturbance signal length
        epsilon: Initial epsilon parameter that determines grid granularity
        epsilon_final: Desired termination epsilon parameter
        lower_bound (numpy array): Lower bounds on the input signal
        upper_bound (numpy array): Upper bounds on the input signal
        checkspec (function): Evaluates whether state trajectory meets a specification
        mine_init_states (bool): If true, will consider initial states as part of assumption space
                                 If false, will only consider disturbance signals

    Optional Args:
        learningrate (double): alpha in paper. Determines change in grid granularity
        max_num_points (int): Miner will terminate after this number of samples
        binary_search (bool): If true, will perform a binary search over the set of epsilons so that
                              the search grid is not unnecessarily fine. A finer grid requires more
                              samples. 
        disturb_over_init_state (double): Used to provide a ratio between bloating in the initial
                                          state direction and the disturbance signal directions

    """
    def __init__(self, system, T, epsilon, epsilon_final,\
                    lower_bound, upper_bound, checkspec,
                    mine_init_states,\
                    order = None, learningrate = .8,\
                    max_num_points = None, transform_u = None, \
                    guarantee = None, binary_search = False,
                    init_state = None,
                    disturb_over_init_state = 1.0,
                    verbose = True): 

        self.T = T
        self.lower_points = np.array([]).reshape((0,T,system.input_dim))
        self.upper_points = np.array([]).reshape((0,T,system.input_dim))

        if mine_init_states:
            self.state_lower_points = np.array([]).reshape((0,system.state_dim))
            self.state_upper_points = np.array([]).reshape((0,system.state_dim))
        else:
            assert (init_state is not None), "Must provide initial state if not mining for them"
            self.init_state = init_state

        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.learningrate = learningrate
        self.checkspec = checkspec
        self.system = system
        self.max_num_points = max_num_points
        self.transform_u = transform_u
        self.mine_init_states = mine_init_states
        self.binary_search = binary_search
        self.verbose = verbose

        if (guarantee is not None) and (hasattr(system,'output_map') is False):
            raise ValueError("Cannot impose guarantee without output_map method")
        else:
            self.guarantee_points = guarantee

        self.solver = z3.Solver()

        self.ndim = self.system.input_dim
        self.input_dim = self.system.input_dim
        #print "Warning, ndim assumes SMT miner only generates inputs"

        try:
            upper_bound[0]
            self.upper_bound = upper_bound
        except:
            self.upper_bound = upper_bound * np.ones([self.ndim])

        try:
            lower_bound[0]
            self.lower_bound = lower_bound
        except:
            self.lower_bound = lower_bound * np.ones([self.ndim])

        # if (self.input_dim > 1) and (len(lower_bound) == 1):
        #     self.lower_bound = lower_bound * np.ones([self.ndim])
        # else:
        #     self.lower_bound = lower_bound

        # if (self.input_dim > 1) and (len(upper_bound) == 1):
        #     self.upper_bound = upper_bound * np.ones([self.ndim])
        # else:
        #     self.upper_bound = upper_bound

        # if order is None:
        #     self.order = np.ones([self.input_dim]) # Default positive order  
        # else:
        #     self.order = order

        # self.signal_order = [order for i in range(self.T)]

        #self.U = [z3.Real("x%s" % i) for i in range(self.T)]

        T = self.T

        self.U = np.array([\
                            [\
                                z3.Real("u%s_%s" % (i,j)) for i in range(self.input_dim)\
                            ]\
                            for j in range(T)\
                          ]\
                         )

        if self.mine_init_states:
            self.z3_x0 = np.array([z3.Real("x%s_0" % i) for i in range(self.system.state_dim)])
            self.impose_state_lower_bound(self.z3_x0, self.system.state_order, self.system.state_lower)
            self.impose_state_upper_bound(self.z3_x0, self.system.state_order, self.system.state_upper)

        self.z3_epsilon = z3.Real('z3_epsilon')
        self.z3_epsilon_state = z3.Real('z3_epsilon_state')
        self.disturb_over_init_state = disturb_over_init_state        
        self.impose_upper_bound(self.U, self.system.input_order)
        self.impose_lower_bound(self.U, self.system.input_order)

        # Keep track of when epsilon is updated
        self.iterations = 0
        self.epsilon_update_iterations = []
        self.epsilon_history = []

    def seed_miner(self):
        """
        Assumption mine on corners of n-dimensional box. 

        Heuristic is geared towards a dynamical system approach. 
        """
        raise NotImplementedError(" ")

    def init_lower_points(self):
        raise NotImplementedError(" ")

    def init_upper_points(self):
        raise NotImplementedError(" ")

    def impose_state_lower_bound(self,z3_var, order, bound = None):
        """
        Impose lower bounds on initial state 
        """
        if bound is None:
            bound = self.lower_bound

        num_inputs = z3_var.shape[-1]

        bound_constraint = [order[i]*z3_var[i] >= order[i]*bound[i]\
                                         for i in range(num_inputs)]
        self.solver.add(bound_constraint)

    def impose_state_upper_bound(self,z3_var, order, bound = None):
        """
        Impose lower bounds on initial state 
        """
        if bound is None:
            bound = self.lower_bound

        num_inputs = z3_var.shape[-1]

        bound_constraint = [order[i]*z3_var[i] <= order[i]*bound[i]\
                                         for i in range(num_inputs)]
        self.solver.add(bound_constraint)

    def impose_lower_bound(self, z3_var, order, bound = None):
        """
        Impose lower bounds on input space for all time
        """
        if bound is None:
            bound = self.lower_bound

        num_inputs = z3_var.shape[-1]

        #num_inputs = self.U.shape[1]
        T = self.U.shape[0]
        assert(T == self.T)
        for t in range(T):

            bound_constraint = [order[i]*z3_var[t,i] >= order[i]*bound[i]\
                                         for i in range(num_inputs)]
            self.solver.add(bound_constraint)

    def impose_upper_bound(self, z3_var, order, bound = None):
        """
        Impose upper bounds on input space for all time
        """
        
        if bound is None:
            bound = self.upper_bound

        #num_inputs = self.U.shape[1]
        num_inputs = z3_var.shape[-1] # last index
        T = self.U.shape[0]
        assert(T == self.T)
        for t in range(T):
            bound_constraint = [order[i]*self.U[t,i] <= order[i]*bound[i]\
                                         for i in range(num_inputs)]
            self.solver.add(bound_constraint)

        #return z3.simplify(rect_disjunction)

    def get_lower_trajectories(self):
        raise NotImplementedError("Relies on depreciated method transform_u")
        lower_inputs = [self.transform_u(x) for x in self.lower_points]
        return np.array([self.system.simulate(x[0], x[1]) for x in lower_inputs])

    def get_upper_trajectories(self):
        raise NotImplementedError("Relies on depreciated method transform_u")
        upper_inputs = [self.transform_u(x) for x in self.upper_points]
        return np.array([self.system.simulate(x[0], x[1]) for x in upper_inputs])

    def get_lower_outputs(self):
        state_traj = self.get_lower_trajectories()
        num_traj = state_traj.shape[0]
        T = state_traj.shape[1]
        return np.array([[self.system.output_map(state_traj[i,t]) for t in range(T)] for i in range(num_traj)])

    def get_upper_outputs(self):
        state_traj = self.get_lower_trajectories()
        num_traj = state_traj.shape[0]
        T = state_traj.shape[1]
        return np.array([[self.system.output_map(state_traj[i,t]) for t in range(T)] for i in range(num_traj)])

    def reset_lower_points(self):
        """
        Method used to reset the set of lower points and epsilon. 

        Primary purpose is to be used in assumption miner
        """
        
        self.epsilon = self.init_epsilon

        self.lower_points = np.array([]).reshape((0,self.T,self.system.input_dim))

    def check_miner(self):
        """
        Double checks that all lower/upper points satisfy a lower spec.
        """
        num_points = self.lower_points.shape[0]
        if self.mine_init_states:
            lower_inputs = [(self.lower_points[i], self.state_lower_points[i]) for i in range(num_points)]
        else:
            lower_inputs = [(x, self.init_state) for x in self.lower_points]
        if (not all([self.checkspec(self.system.simulate(x[0], x[1])) for x in lower_inputs])):
            return False

        num_points = self.upper_points.shape[0]            
        if self.mine_init_states:
            upper_inputs = [(self.upper_points[i], self.state_upper_points[i]) for i in range(num_points)]
        else: 
            upper_inputs = [(x, self.init_state) for x in self.upper_points]
        if not all([not self.checkspec(self.system.simulate(x[0], x[1])) for x in upper_inputs]):
            return False

        return True 

    def check_guarantee(self, traj, guar_type = 'state'):
        """
        Checks if a state trajectory (traj) is dominated by at least one point.

        If guar_type is 'output', then automatically translates state trajectory
        to output trajectory.

        """            

        if not (guar_type == 'state' or guar_type == 'output'):
            raise ValueError("Guarantee must be over state or output trajectories")

        num_points = self.guarantee_points.shape[0]

        def sig_dom(x,y,o):
            """
            Determines if signal x is dominated by y for all time with order o. 

            x,y have dimensions [signal length, dimension of valuation (e.g. 2 for R^2)]
            """

            assert(x.ndim == 2 and y.ndim == 2)
            T = x.shape[0]
            dim = x.shape[1]
            assert(T == y.shape[0]), "Signals must be of same length"
            assert(dim == y.shape[1]), "Signals must be of same length"

            return (x * o <= y * o).all() # elementwise multiplication

        if guar_type == 'output': # map state trajectory to outputs
            test_traj = np.array([self.system.output_map(traj[t]) for t in range(T)])
            order = self.system.output_order
            return any([sig_dom(test_traj, self.guarantee_points[i], order) for i in range(num_points)])
        else:
            test_traj = traj # new test_traj declared because traj passed by reference
            order = self.system.state_order
            return any([sig_dom(test_traj, self.guarantee_points[i], order) for i in range(num_points)])

    def simplify_set_underapprox(self, points, set_type):
        """
        Takes a set of points under-approximating a lower or an upper set.
        Eliminates redundancies in the sets
        """

        if set_type != 'upper' and set_type != 'lower':
            raise ValueError(" ")

        print "Set underapproximations are restricted to regular coordinate-wise order."

        #points = np.flipud(points) # boundary points usually near the end

        num_points = points.shape[0]
        index = np.ones(num_points, dtype = bool) # indices to keep 

        for i in range(num_points):
            if set_type == 'upper':
                b_eq = (points > points[i]) # dominated points
            elif set_type == 'lower':
                b_eq = (points < points[i]) # dominated points 

            # Calculate dominated points
            dominated_index = np.array([b_eq[i].all() for i in range(num_points)], dtype = bool)
            not_dominated = np.logical_not(dominated_index)
            index = np.logical_and(index, not_dominated)

        return index

    def clean_constraints(self):
        """ 
        Checks for redundant sample points and gets rid of them if found
        """
        self.solver.reset()

        self.impose_upper_bound(self.U, self.system.input_order)
        self.impose_lower_bound(self.U, self.system.input_order)

        lower_index = self.simplify_set_underapprox(self.lower_points, 'lower')
        upper_index = self.simplify_set_underapprox(self.upper_points, 'upper')

        if self.mine_init_states:
            self.z3_x0 = np.array([z3.Real("x%s_0" % i) for i in range(self.system.state_dim)])
            self.impose_state_lower_bound(self.z3_x0, self.system.state_order, self.system.state_lower)
            self.impose_state_upper_bound(self.z3_x0, self.system.state_order, self.system.state_upper)

            lower_index = np.logical_and(lower_index, simplify_set_underapprox(self.state_lower_points, 'lower'))
            upper_index = np.logical_and(upper_index, simplify_set_underapprox(self.state_upper_points, 'upper'))

            self.state_lower_points = self.state_lower_points[lower_index]
            self.state_upper_points = self.state_upper_points[upper_index]

        self.lower_points = self.lower_points[lower_index]
        self.upper_points = self.upper_points[upper_index]

        # Reapply lower/upper constraints 
        raise NotImplementedError("Need to reapply constraints")
        rect_disjunction = []
        for i in self.lower_points.shape[0]:
            for t in range(self.T):
                input_disjunction = not_in_rectangle(self.U[t,:], self.lower_points[i,t], \
                                                        'lower', self.z3_epsilon, input_order)
                rect_disjunction.append(input_disjunction)

            if self.mine_init_states:
                state_disjunction = not_in_rectangle(self.z3_x0, self.state_lower_points[i], 'lower',\
                                            self.z3_epsilon, state_order)
                rect_disjunction.append(state_disjunction)

        for i in self.upper_points.shape[0]: 
            for t in range(self.T):
                input_disjunction = not_in_rectangle(self.U[t,:], self.upper_points[i,t], \
                                                        'upper', self.z3_epsilon, input_order)
                rect_disjunction.append(input_disjunction)

            if self.mine_init_states: 
                state_disjunction = not_in_rectangle(self.z3_x0, self.state_upper_points[i], 'upper',\
                                                        self.z3_epsilon, state_order)
                rect_disjunction.append(state_disjunction)

        self.solver.add(z3.Or(rect_disjunction))

    def call_solver(self, epsilon):
        self.solver.push()
        self.solver.add(self.z3_epsilon == self.epsilon * self.disturb_over_init_state)
        self.solver.add(self.z3_epsilon_state == self.epsilon)
        self.solver.pop()

        return (satisfied == z3.sat)

    def mine(self):
        """
        Mines assumption points until epsilon reaches epsilon_final
        Stores the sampled points in lower_points, upper_points
        """
        input_order = self.system.input_order
        state_order = self.system.state_order

        while(self.epsilon >= self.epsilon_final):

            num_lower = self.lower_points.shape[0]
            num_upper = self.upper_points.shape[0]
            total_points = num_lower + num_upper
            if self.verbose and (total_points % 500 == 0):
                print "Points: ", total_points, ", epsilon: ", self.epsilon
            
            if (self.max_num_points is not None) and (total_points >= self.max_num_points):
                # TODO: Prune points that are dominated and double check
                break

            self.solver.push()
            self.solver.add(self.z3_epsilon == self.epsilon * self.disturb_over_init_state) # Input constraint 
            self.solver.add(self.z3_epsilon_state == self.epsilon) # State constraint 
            satisfied = self.solver.check()
            self.solver.pop()

            self.iterations += 1
            self.epsilon_history.append(self.epsilon)

            if (satisfied == z3.sat):
                # Extract point
                model = self.solver.model()

                # TODO change this to account for multidimension signals
                if self.mine_init_states:
                    new_sample_u = np.array([\
                                                [z3_to_float(model[self.U[t,i]]) for i in range(self.input_dim)]\
                                                for t in range(self.T)\
                                          ])
                    new_sample_x0 = np.array([z3_to_float(model[self.z3_x0[i]]) for i in range(self.system.state_dim)])
                else: 
                    new_sample_u = np.array([\
                                                [z3_to_float(model[self.U[t,i]]) for i in range(self.input_dim)]\
                                                for t in range(self.T)\
                                          ])
                    #new_sample_u, new_sample_x0 = self.transform_u(new_sample)
                    new_sample_x0 = self.init_state

                sample_trajectory = self.system.simulate(new_sample_u, new_sample_x0)

                # Determine in upper/lower set and add constraint
                rect_disjunction = []

                requirement_met = True # requirement_met = spec_met & guarantee_met
                if self.guarantee_points is not None:
                    requirement_met &= self.check_guarantee(sample_trajectory, guar_type = 'output')
                requirement_met &= self.checkspec(sample_trajectory)

                if requirement_met:
                    self.lower_points = np.vstack((self.lower_points, np.expand_dims(new_sample_u, 0)))
                    for t in range(self.T):
                        input_disjunction = not_in_rectangle(self.U[t,:], new_sample_u[t], \
                                                                'lower', self.z3_epsilon, input_order)
                        rect_disjunction.append(input_disjunction)

                    if self.mine_init_states:
                        self.state_lower_points = np.vstack((self.state_lower_points, new_sample_x0))
                        state_disjunction = not_in_rectangle(self.z3_x0, new_sample_x0, 'lower',\
                                                                self.z3_epsilon_state, state_order)
                        rect_disjunction.append(state_disjunction)
                    
                else:
                    self.upper_points = np.vstack((self.upper_points, np.expand_dims(new_sample_u, 0)))
                    for t in range(self.T):
                        input_disjunction = not_in_rectangle(self.U[t,:], new_sample_u[t], \
                                                                'upper', self.z3_epsilon, input_order)
                        rect_disjunction.append(input_disjunction)

                    if self.mine_init_states:
                        self.state_upper_points = np.vstack((self.state_upper_points, new_sample_x0))
                        state_disjunction = not_in_rectangle(self.z3_x0, new_sample_x0, 'upper',\
                                                                self.z3_epsilon_state, state_order)
                        rect_disjunction.append(state_disjunction)

                self.solver.add(z3.Or(rect_disjunction)) # Or b/c only need to be greater at single t

                
            else: # z3_epsilon is too big. shrink it
                #binary_search = False
                self.prev_epsilon = self.epsilon
                self.epsilon *= self.learningrate
                #max_space = max(self.upper_bound - self.lower_bound)
                #self.epsilon *= ( (max_space - self.epsilon) / max_space)

                # Search until epsilon barely gives sat
                if self.binary_search:
                    if (self.call_solver(self.epsilon)): # next iteration isn't bloated enough 
                        upper_epsilon = self.prev_epsilon
                        lower_epsilon = self.epsilon
                        precision = upper_epsilon / 10.
                        while (lower_epsilon + precision < upper_epsilon):
                            midpoint = ((upper_epsilon + lower_epsilon) / 2.0)
                            if(self.call_solver(midpoint)):
                                lower_epsilon = midpoint
                                self.epsilon = lower_epsilon
                            else: 
                                upper_epsilon = midpoint
                        self.prev_epsilon = upper_epsilon
                    else:
                        pass

                self.epsilon_update_iterations.append(self.iterations)

        print "Miner terminated" 


class guarantee_certifier(object):
    """
    z3dynamics:
    lower_points:
    guarantee: Lower points of other system 
    """


    def __init__(self, system, assumption, guarantee_type, guarantee, T = None):

        if type(assumption) is types.ListType:
            assumption = np.array(assumption)

        if type(guarantee) is types.ListType:
            guarantee = np.array(guarantee)

        assert(type(guarantee) == type(np.array([1])))
        assert(type(assumption) == type(np.array([1])))

        self.system = system
        self.z3dynamics = system.z3_dynamics_constraints
        self.lower_points = assumption
        self.upper_points = None
        self.guarantee = guarantee

        self.state_dim = system.state_dim
        self.input_dim = system.input_dim
        self.output_dim = system.output_dim

        assert(system.state_order.size == system.state_dim)
        assert(system.input_order.size == system.input_dim)

        if not (guarantee_type == 'state' or guarantee_type == 'output'):
            raise ValueError('guarantee type must be over states or outputs')
        self.guarantee_type = guarantee_type

        if T is not None:
            assert (self.lower_points.shape[1] == T), "Assumption signals must be same length as time T"
            assert (self.lower_points.shape[1] == T), "Assumption signals must be same length as time T"
            self.T = T
        else:
            self.T = self.lower_points.shape[1] 

        self.solver = z3.Solver()

        dyn_constraints, self.z3x, self.z3u = self.z3dynamics(self.T)

        assert (system.state_dim == self.z3x.shape[1]), "z3 state variable dimension inconsistent"
        assert (system.input_dim == self.z3u.shape[1]), "z3 input variable dimension inconsistent"

        #z3y = np.array([[z3.Real("y%i_%i" % (j,t)) for j in range(self.output_dim)] for t in range(self.T)])
        output_constraints, self.z3y = self.system.z3_output_constraints(self.z3x)

        self.solver.add(dyn_constraints)
        self.solver.add(output_constraints)

    def z3_true_assumption(self, z3_u):
        """
        Impose that input signal u[] must be in lower set.

        Or over lower points
        And over time
        And over dimensions 
        """

        order = self.system.input_order
        constraints = []
        num_points = self.lower_points.shape[0]
        for i in range(num_points):
            temp = []
            for t in range(self.T):
                temp.append(in_rectangle(z3_u[t], self.lower_points[i,t], 'lower', 0.0, order))
            constraints.append(z3.And(temp))

        return z3.simplify(z3.Or(constraints))

    def z3_false_guarantee(self, z3_var):
        """
        Impose that state signal x[] not be in a lower set represented 
        with maximal points.

        In order to falsify, x must dominate or be incomparable to every point in lower set
        And over lower points
        exist a time for which
        exist a dimension that exceeds
        """

        order = self.system.input_order
        constraints = []
        num_points = self.guarantee.shape[0]
        for i in range(num_points):
            temp = []
            for t in range(self.T):
                temp.append(not_in_rectangle(z3_var[t], self.guarantee[i,t], 'lower', 0.0, order))
            constraints.append(z3.Or(temp))

        return z3.simplify(z3.And(constraints))

    def falsify(self):
        """
        Attempts to falsify guarantee, given the dynamics and assumptions

        Can be called repeatedly with different sets of assumptions and 
        guarantees because they are generated in this method. 
        """

        self.solver.push()

        AGconstraints = []

        AGconstraints.append(self.z3_true_assumption(self.z3u))
        if self.guarantee_type == 'state':
            AGconstraints.append(self.z3_false_guarantee(self.z3x))
        elif self.guarantee_type == 'output':
            AGconstraints.append(self.z3_false_guarantee(self.z3y))
        self.solver.add(AGconstraints)

        satisfied = self.solver.check()

        self.solver.pop() 

        if (satisfied == z3.sat):
            self.z3model = self.solver.model()
            return True
        else:
            self.z3model = None
            return False


class AG_miner(object):
    def __init__(self, amine_1, gfal_1, amine_2, gfal_2):
        assert(gfal_1.guarantee_type == 'output')
        assert(gfal_2.guarantee_type == 'output')
        self.amine_1 = amine_1
        self.gfal_1 = gfal_1

        self.amine_2 = amine_2
        self.gfal_2 = gfal_2

    def mineAG(self):
        """
        
        """

        falsified_1 = True
        falsified_2 = True 
        while (falsified_1 or falsified_2):
            if falsified_1:
                # Reset miner
                self.amine_1.reset_lower_points()
                self.amine_1.epsilon = self.amine_1.init_epsilon

                self.amine_1.mine()

            if falsified_2:
                # Reset miner
                self.amine_2.reset_lower_points()
                self.amine_2.epsilon = self.amine_2.init_epsilon

                self.amine_2.mine()

            self.gfal_1.guarantee = self.amine_2.lower_points
            self.gfal_2.guarantee = self.amine_1.lower_points

            self.gfal_1.lower_points = self.amine_1.lower_points
            self.gfal_2.lower_points = self.amine_2.lower_points

            # guar_1 = assum_2 
            # guar_2 = assum_1

            while(True):
                falsified_1 = self.gfal_1.falsify()

                if falsified_1 is not False:
                    # Extract trajectory from counterexample and check if spurious. 

                    z3x = gfal_1.z3x
                    T = z3x.shape[0]
                    xdim = z3x.shape[1]
                    out_trajectory_1 = np.array([[gfal_1.z3model[z3x[t,i]] for i in range(xdim)] for t in range(T)])
                    sample_trajectory_2 = self.amine_2.system.simulate(out_trajectory_1, new_sample_x0)
                    # 
                    if (amin_2.checkspec(sample_trajectory_2)):
                        raise NotImplementedError("Add spurious output trajectory to assum_2, guar_1")
                        amine_2.lower_points = np.vstack((amine_2.lower_points, np.expand_dims(out_trajectory_1,0)))
                        gfal_1.guarantee = np.vstack((gfal_1.guarantee, np.expand_dims(out_trajectory_1,0)))
                        print("WARNING: GFAL FALSIFIES OVER STATE TRAJECTORIES. Fix above line to be over outputs")
                        continue
                    else:
                        break
                        # true counterexample found. need to assum mine 1 with new guarantee
                else:
                    # failed to falsify. guarantee met
                    break 

            while(True):
                falsified_2 = self.gfal_2.falsify()

                if falsified_2 is not False:
                    # Extract trajectory from counterexample and check if spurious. 
                    z3x = gfal_1.z3x
                    T = z3x.shape[0]
                    xdim = z3x.shape[1]

                    # Get System 1's behavior given system 2's input to it.
                    out_trajectory_2 = np.array([[gfal_1.z3model[z3x[t,i]] for i in range(xdim)] for t in range(T)])
                    sample_trajectory_1 = self.amine_1.system.simulate(out_trajectory_2, new_sample_x0)
                    
                    if (amin_1.checkspec(sample_trajectory_1)):
                        raise NotImplementedError("Add spurious output trajectory to assum_1, guar_2")
                        amine_1.lower_points = np.vstack((amine_1.lower_points, np.expand_dims(out_trajectory_2,0)))
                        gfal_2.guarantee = np.vstack((gfal_2.guarantee, np.expand_dims(out_trajectory_2,0)))
                        print("WARNING: GFAL FALSIFIES OVER STATE TRAJECTORIES. Fix above line to be over outputs")
                        continue
                    else:
                        break
                        # true counterexample found. need to assum mine 2 with new guarantee
                else:
                    # failed to falsify. guarantee met
                    break 

            if falsified_1 and falsified_2:
                falsified_2 = False

        return self



