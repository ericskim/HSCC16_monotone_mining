import re 
import types

import numpy as np

class STL(object):
    """
    A signal temporal logic STL object that can be called with a signal to evaluate 
    satisfaction of the specification. 

    Optional Args:
        stl_type (str): Gives the 
        pred_eval: If the stl_type == 'pred', a function that determines 
                   satisfaction of the signal is required. 
        phi (STL): If stl_type is 'not', 'always', 'eventually' then this is the sub-specification
        phi1 (STL): If stl_type is 'or', 'and', 'until' then this the first sub-specification
        phi2 (STL): If stl_type is 'or', 'and', 'until' then this the second sub-specification
        a (int): 
        b (int): 



    """

    def __init__(self, stl_type = 'pred', pred_eval = None,\
                 phi = None, phi1 = None, phi2 = None, a = None, b = None):

        self.stl_type = stl_type
        self.a = a
        self.b = b

        if stl_type is 'pred':
            assert(pred_eval is not None), 'Predicate should not be empty'
            assert(hasattr(pred_eval, '__call__')), 'Predicate must be callable'
            assert(phi2 is None)
            self.phi1 = None
            self.phi2 = None
            self.stl_type = 'pred'
            self.pred_eval = pred_eval
            
        elif stl_type is 'or':
            assert(phi2 is not None)
            self.phi1 = phi1
            self.phi2 = phi2
            
        elif stl_type is 'and':
            assert(phi2 is not None)
            self.phi1 = phi1
            self.phi2 = phi2

        elif stl_type is 'always':
            self.phi = phi

        elif stl_type is 'eventually': 
            self.phi = phi

        elif stl_type is 'until':
            self.phi1 = phi1
            self.phi2 = phi2

        elif stl_type is 'not':
            self.phi = phi
        else:
            raise ValueError('Not valid stl formula type')

    def __call__(self, x):

        err_b_leq_a = "Upper bound must be greater than lower bound"
        
        if self.stl_type is 'pred':
            assert(self.pred_eval is not None), 'Predicate should not be empty'
            #assert(y is None)
            
            out = self.pred_eval(x)
            assert(out.dtype == types.BooleanType)
            return out
        
        elif self.stl_type is 'or':
            assert(phi1 is not None)
            assert(phi2 is not None)
            return np.logical_or(self.phi1(x), self.phi2(x))

        elif self.stl_type is 'and':
            assert(phi1 is not None)
            assert(phi2 is not None)
            return np.logical_and(self.phi1(x), self.phi2(x))

        elif self.stl_type is 'always':
            T = x.shape[0]
            if (self.a is not None) and (self.b is not None):
                assert(self.a <= self.b), err_b_leq_a
                a = self.a
                b = self.b
            elif (self.a is None and self.b is None):
                a = 0
                b = T # maximum time interval
            else:
                raise ValueError("Cannot have only one bound defined.")
                
            x = self.phi(x)
            return np.array([(x[t+a:min(t+b+1,T)]).all() for t in range(T)])
            #return np.array([self.phi(x[t+a:min(t+b,T)]).all() for t in range(T)])
            
                
        elif self.stl_type is 'eventually': 
            T = x.shape[0]
            if (self.a is not None) and (self.b is not None):
                assert(self.a <= self.b), err_b_leq_a
                a = self.a
                b = self.b
            elif (self.a is None and self.b is None):
                a = 0
                b = T # maximum time interval
            else:
                raise ValueError("Cannot have only one bound defined.")
                
            x = self.phi(x)
            return np.array([(x[t+a:min(t+b+1,T)]).any() for t in range(T)])
            #return np.array([self.phi(x[t+a:min(t+b,T)]).any() for t in range(T)])

        elif self.stl_type is 'until':
            T = x.shape[0]
            
            if (self.a is not None) and (self.b is not None):
                assert(self.a <= self.b), err_b_leq_a
                a = self.a
                b = self.b                
            elif (self.a is None and self.b is None):
                a = 0
                b = T # maximum time interval
            else:
                raise ValueError("Cannot have only one bound defined.")
            x1 = self.phi1(x)
            x2 = self.phi2(x)
            
            ev_phi2 = np.array([(x2[min(t+a,T-1):min(t+b+1,T)]).any() for t in range(T)])
            
            def ev(y):
                """
                Find index of first time x is true
                """
                try:
                    return np.where(y)[0][0]
                except:
                    return None

            def alw(y, a, b):
                """
                True if y is true between a and b indices 
                """
                if (b is not None):

                    return y[min(i+a,T-1):min(i+b,T)].all()
                else: 
                    return False


            ev_index = np.array([ev(x2[min(i+a,T-1):min(i+b+1, T)]) for i in range(T)]) # index first time x2 true
            
            true_until_index = np.array([alw(x1,a,ev_index[i]) for i in range(T)])
            
            return np.logical_and(ev_phi2, true_until_index)

        elif self.stl_type is 'not':
            return np.logical_not(self.phi(x))
            
    def until(self, a = None, b = None, phi2 = None):
        """
        Create a new STL specification from other others
        phi1 is true until phi2 is true, and phi2 must become true

        Example usage:
            def get_true(x):
                return True
            other_phi = STL(pred_eval = get_true)
            phi1.until(phi2 = other_phi, a = 0, b = 3)
        """
        assert(phi2 is not None)
        return STL(phi1 = self, phi2 = phi2, stl_type = 'until', a = a, b = b)
            
    def __or__(self, other):
        if self.stl_type is 'pred' and other.stl_type is 'pred':
            new_pred = lambda x: self.pred_eval(x) | other.pred_eval(x)
            return STL(stl_type = 'pred', pred_eval = new_pred)
        else: 
            return STL(phi1 = self, phi2 = other, stl_type = 'or')

    def __and__(self, other):
        if self.stl_type is 'pred' and other.stl_type is 'pred':
            new_pred = lambda x: self.pred_eval(x) & other.pred_eval(x)
            return STL(stl_type = 'pred', pred_eval = new_pred)
        else: 
            return STL(phi1 = self, phi2 = other, stl_type = 'and')

    def __invert__(self):
        return STL(phi = self, stl_type = 'not')

def G(phi, a = None, b = None):
    """
    phi must be true always in [a,b]
    """
    return STL(phi = phi, stl_type = 'always', a = a, b = b)
    
def F(phi, a = None, b = None):
    """
    phi must be true some time in [a,b]
    """
    return STL(phi = phi, stl_type = 'eventually', a = a, b = b)
    
def until(phi1, phi2, a = None, b = None):
    """
    phi1 must be true until phi2 is true
    """
    return STL(phi1 = phi1, phi2 = phi2, stl_type = 'until', a = a, b = b)
