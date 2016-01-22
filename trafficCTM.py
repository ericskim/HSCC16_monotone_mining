"""
Cell Transmission Model Traffic Example Class 

"""

import types
import numpy as np 

import z3


class actm(object):
    def __init__(self):

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
        """
        Simulate CTM Network dynamics

        Args:
            u (2D numpy array): Input signal with array dimensions (signal length, signal dimension)
            x0 (numpy array): Initial state of network. Must be horizontal 1D array. 
        """

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
        Check if state signal x doesn't exhibit congestion for all time
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
        Return a signal determining if congestion free exists at any given time.
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


