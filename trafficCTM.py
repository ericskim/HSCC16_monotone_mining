import numpy as np 
import z3
import types

class actm(object):
    def __init__(self):
        #self.x0 = x0
        self.state_dim = 5
        self.input_dim = 2
        self.output_dim = 1
        self.state_order = np.ones([self.state_dim])
        self.input_order = np.ones([self.input_dim])
        self.output_order = np.ones([self.output_dim])
        self.state_lower = np.array([0,0,0,0,0])
        self.state_upper = np.array([1,1,1,.5,.5])
        self.input_lower = np.array([0,0])
        self.input_upper = np.array([.3,.3])
        self.road_saturation = .4
        self.ramp_saturation = .3

    def simulate(self, u, x0):
        T = u.shape[0]
        x = np.zeros([T+1, self.state_dim])
        x[0] = x0

        segmax = [1,1,1, .5, .5 ]

        s = .5
        d = .3

        for t in range(T):

            # Flows f_ij from link i to j
            f_01 = min(.8*(segmax[1] - x[t, 1]), self.road_saturation, .5 * x[t, 0])
            f_12 = min(.8*(segmax[2] - x[t, 2]), self.road_saturation, .5 * x[t, 1])
            f_2 = min(s, self.road_saturation, .5 * x[t, 2]) # out of network

            # Ramp flows
            f_r3 = min(.2*(segmax[1] - x[t, 1]), self.ramp_saturation, x[t, 3])
            f_r4 = min(.2*(segmax[2] - x[t, 2]), self.ramp_saturation, x[t, 4])

            # Segment Updates
            x[t,0] = min(segmax[0], x[t, 0] + d - f_01)
            x[t,1] = min(segmax[1], x[t, 1] - f_12 + f_r3)
            x[t,2] = min(segmax[2], x[t, 2] - f_2 + f_r4)

            # Ramp updates 
            x[t,3] = min(segmax[3], x[t, 3] - f_r3 + u[t, 0])
            x[t,4] = min(segmax[4], x[t, 4] - f_r4 + u[t, 1])

        return x

    def congestion_free(self, x):
        """
        Check if state signal x doesn't exhibit congestion
        """

        T = x.shape[0]
        segmax = [1, 1, 1, .5, .5]
        no_congestion = True

        for t in range(T):
            if ((.5 < .5 * x[t, 2]) \
                    or (.8*(segmax[1] - x[t, 1]) < min(self.road_saturation, .5 * x[t, 0])) \
                    or (.8*(segmax[2] - x[t, 2]) < min(self.road_saturation, .5 * x[t, 1])) \
                    or (.2*(segmax[1] - x[t, 1]) < min(self.ramp_saturation, x[t, 3]))     \
                    or (.2*(segmax[2] - x[t, 2]) < min(self.ramp_saturation, x[t, 4]))):

                no_congestion = False
                break

        return no_congestion 

    def congestion_signal(self, x):
        """
        Return a time signal determining if congestion free exists over time.
        """
        T = x.shape[0]
        segmax = [1, 1, 1, .5, .5]
        congestion_signal = np.zeros([T], dtype=bool)

        for t in range(T):
            if ((.5 < .5 * x[t, 2]) \
                    or (.8*(segmax[1] - x[t, 1]) < min(self.road_saturation, .5 * x[t, 0])) \
                    or (.8*(segmax[2] - x[t, 2]) < min(self.road_saturation, .5 * x[t, 1])) \
                    or (.2*(segmax[1] - x[t, 1]) < min(self.ramp_saturation, x[t, 3]))     \
                    or (.2*(segmax[2] - x[t, 2]) < min(self.ramp_saturation, x[t, 4])) ):

                congestion_signal[t] = False # congestion
            else:
                congestion_signal[t] = True # no congestion

        return congestion_signal 

# class actm(object):
#     def __init__(self):
#         """
#         Gives a set of default parameters. 
#         """
#         self.nSegments = 3
#         self.has_onramp = np.array([1,1,0])
#         self.has_offramp = [1,0,0]
#         self.is_metered = [1,1,0]
#         self.xMax = [11.5, 11.5, 11.5]
#         self.x0 = [0.0] * self.nSegments
#         self.gamma = [1.0] * self.nSegments
#         self.seg_Fbar = [2, 2, 2]
#         self.freeflow_velocity = [0.7241, 0.7241, 0.7241]
#         self.beta = np.array([.1, 0,0])
#         self.beta_bar = 1 - self.beta

#         self.umin = [.2,.2,0];
#         self.umax = [.5,.5,0];
      
#         self.segD = [2, 0, 0];
#         self.onrampD = [0.6, .6, 0];

#         self.Xi = np.ones(self.nSegments);

#         self.w = .181 * np.ones(self.nSegments);

#         self.L = 8; 
#         self.ts = 3;
#         self.control_type = 'alinea'

#     def simulate(self, u, x0):

#         assert(isinstance(u, types.TupleType))
#         assert(len(u) == 2)
#         d = u[0]
#         s = u[1]

#         assert (self.control_type is 'alinea' or self.control_type is 'fixed')
#         assert (d.shape[0] == s.shape[0]), "Inputs must have same time horizon"# same time horizon for supply/demand
#         assert (x0.size == self.nSegments * 2), "Initial state has incorrect length"
#         #assert (d.shape[1] == 1  + num_onramps), "Demand should only be for first segment and onramps"
#         assert (s.shape[1] == 1), "Supply should only exist for last freeway segment"

#         T = d.shape[0] + 1 
        
#         nramps = np.count_nonzero(self.has_onramp)
#         state_dimension = self.nSegments + nramps

#         x = np.zeros([T, self.nSegments * 2])
#         x[0] = x0
#         demand   = np.zeros([T, self.nSegments])
#         supply   = np.zeros([T, self.nSegments])
        
#         segflow  = np.zeros([T, self.nSegments])
#         onrampflow = np.zeros([T, self.nSegments])
#         offrampflow = np.zeros([T, self.nSegments])

#         for t in range(T):
#             for seg in range(self.nSegments):

#                 x_ramp = seg + self.nSegments # this convention is confusing

#                 demand[t,seg] = min([x[t,seg], self.seg_Fbar[seg]])
#                 supply[t,seg] = self.xMax[seg] - x[t,seg]

#                 if has_onramp[seg]:
#                     ramp_demand = x[t, x_ramp] + d[t,x_ramp]
#                     onrampflow[t, seg] = min(ramp_demand, self.Xi[seg] * supply[t,seg])
#                 else:
#                     pass

#                 if has_offramp[seg]:
#                     pass

#                 if (seg == self.nSegments - 1): # No downstream segment exists
#                     pass
#                 else: # Downstream segment exists
#                     pass


#                 if (seg == 0):
#                     pass
#                     # x[t + 1, seg] = 
#                 else:
#                     pass

#         return x

#     def z3Constraints(self, x, d, s):
#         """
#         Parameters:

#         Output:
#         """
#         def z3max(assignment, array):
#             assert (isinstance(array,types.TupleType) or isinstance(array,types.ListType))

#             constraints = [assignment >= array[i] for i in range(len(array))]
#             equal_to_constraint = z3.Or([assignment == array[i] for i in range(len(array))])
#             constraints.append(equal_to_constraint)
            
#             return constraints

#         def z3min(assignment, array):
#             assert (isinstance(array,types.TupleType) or isinstance(array,types.ListType))

#             constraints = [assignment <= array[i] for i in range(len(array))]
#             equal_to_constraint = z3.Or([assignment == array[i] for i in range(len(array))])
#             constraints.append(equal_to_constraint)

#             return constraints
