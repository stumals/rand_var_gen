from scipy.stats import chisquare, shapiro
import numpy as np
import math

from random_var import ContinuousRV, DiscreteRV

class TestDiscreteRV():
    def __init__(self):
        self.discrete = DiscreteRV()

    def test_binomial(self, p=0.5, size=10000):
        '''
        Chi-square test
        '''
        rv = self.discrete.binomial(p=p, size=size)
        f_obs = np.unique(rv, return_counts=True)[1]
        f_exp = np.array([1-p, p])*size
        return chisquare(f_obs, f_exp)
    
    def test_geometric(self, p=0.5, size=10000):
        '''
        Chi-square test
        '''
        rv = self.discrete.geometric(p=p, size=size)
        f_obs = np.unique(rv, return_counts=True)[1]
        f_exp = size*(1-p)**np.arange(0, len(f_obs))*p
        f_exp[-1] = f_exp[-1] + (size - f_exp.sum())
        return chisquare(f_obs, f_exp)
    
    def test_poisson(self, lamb=1, size=10000):
        '''
        Chi-square test
        '''
        rv = self.discrete.poisson(lamb=lamb, size=size)
        f_obs = np.unique(rv, return_counts=True)[1]
        probs = []
        for k in np.arange(0,len(f_obs)):
            pmf = ((lamb**k)*np.exp(-lamb))/math.factorial(k)
            print(k, pmf)
            probs.append(size*pmf)
        f_exp = np.array(probs)
        f_exp[-1] = f_exp[-1] + (size - f_exp.sum())
        return chisquare(f_obs, f_exp)
    
    def test_discrete_rv(self, probabilities=[0.2, 0.3, 0.5], values=[1, 2, 3], size=10000):
        '''
        Chi-square test
        '''
        rv = self.discrete.discrete_rv(probabilities=probabilities, values=values, size=size)
        f_obs = np.unique(rv, return_counts=True)[1]
        f_exp = np.array(probabilities)*size
        return chisquare(f_obs, f_exp)
    


class TestContinuousRV():
    def __init__(self):
        self.continuous = ContinuousRV()

    def test_uniform(self, a=0, b=1, size=10000, bins=10):
        '''
        Chi-square test
        '''
        f_obs = self.continuous.uniform(a=a, b=b, size=size)
        bin_array = np.linspace(a, b, bins+1)
        f_obs = np.unique(np.digitize(f_obs, bin_array), return_counts=True)[1]
        f_exp = np.array([size/bins]*bins)
        return chisquare(f_obs, f_exp)

    def test_normal(self, mu=0, sigma=1, size=1000):
        '''
        Shapiro-Wilk test
        '''
        f_obs = self.continuous.normal(mu=mu, sigma=sigma, size=size)
        return shapiro(f_obs)
    
    def test_exponential(self, lamb=1, size=10000, bins=10):
        '''
        Chi-square test
        '''
        f_obs = self.continuous.exponential(lamb=lamb, size=size)
        a = []
        for i in np.arange(0,bins):
            a.append((-1/lamb)*np.log(1-(i/bins)))
        a.append(np.inf)
        f_obs = np.unique(np.digitize(f_obs, a), return_counts=True)[1]
        f_exp = np.array([size/bins]*bins)
        return chisquare(f_obs, f_exp)
    
    def test_weibull(self, lamb=1, b=1, size=10000, bins=10):
        '''
        Chi-square test
        '''
        f_obs = self.continuous.weibull(lamb=lamb, b=b, size=size)
        a = []
        for i in np.arange(0,bins):
            a.append((1/lamb)*(-np.log(1-(i/bins)))**(1/b))
        a.append(np.inf)
        f_obs = np.unique(np.digitize(f_obs, a), return_counts=True)[1]
        f_exp = np.array([size/bins]*bins)
        return chisquare(f_obs, f_exp)
    
    def test_triangular(self, a=0, b=1, c=.5, size=10000, bins=10):
        '''
        Chi-square test
        '''
        rv = self.continuous.triangular(a=a, b=b, c=c, size=size)
        f_obs = np.unique(np.digitize(rv, np.linspace(a, b, bins+1)), return_counts=True)[1]
        tri_cdf = lambda x, a, b, c: (x-a)**2/((b-a)*(c-a)) if a <= x < c else 1 - (b-x)**2/((b-a)*(b-c))
        probs = []
        for x in np.linspace(a, b, bins+1):
            probs.append(tri_cdf(x, a, b, c))
        f_exp = np.ediff1d(np.array(probs))*size
        return chisquare(f_obs, f_exp)
