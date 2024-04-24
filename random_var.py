#%%
import numpy as np
import matplotlib.pyplot as plt
# %%

class RV():
    def __init__(self, seed=42):
        '''
        set seed for reproducibility
        '''
        np.random.seed(seed)

    def plot_dist(self, rv, dist_name, plot_comparison=False, bins=30):
        '''
        Plot random variable distribution

        rv: random variable array
        dist_name: distribution name
        plot_comparison: plot comparison with numpy distribution
        bins: number of bins for histogram
        '''
        if plot_comparison:
            if dist_name == 'Binomial':
                distribution = np.random.binomial(1, self.p, self.size),
            elif dist_name == 'Geometric':
                distribution = np.random.geometric(self.p, self.size),
            elif dist_name == 'Poisson':
                distribution = np.random.poisson(self.lamb, self.size),
            elif dist_name == 'Discrete':
                distribution = np.random.choice(self.values, self.size, p=self.probabilities),
            elif dist_name == 'Uniform':
                distribution = np.random.uniform(self.a, self.b, self.size),
            elif dist_name == 'Exponential':
                distribution = np.random.exponential(self.lamb, self.size),
            elif dist_name == 'Weibull':
                distribution = np.random.weibull(self.b, self.size),
            elif dist_name == 'Normal':
                distribution = np.random.normal(self.mu, self.sigma, self.size),
            elif dist_name == 'Triangular':
                distribution = np.random.triangular(self.a, self.c, self.b, self.size),
            elif dist_name == 'Chi-Square':
                distribution = np.random.chisquare(self.n, self.size)
            else:
                raise ValueError('Invalid distribution name')
        
        if not plot_comparison:
            fig, ax = plt.subplots()
            ax.hist(rv, bins=bins)
            n, bins, patches = ax.hist(rv, bins=bins, color='blue', edgecolor='black', linewidth=1.2)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            ax.plot(bin_centers, n, color='r', linestyle='dashed')
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{dist_name} Distribution')

        else:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            #ax[0].hist(rv, bins=bins)
            n, bins, patches = ax[0].hist(rv, bins=bins, color='blue', edgecolor='black', linewidth=1.2)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            ax[0].plot(bin_centers, n, color='r', linestyle='dashed')
            ax[0].set_xlabel('Value')
            ax[0].set_ylabel('Frequency')
            ax[0].set_title(f'{dist_name} Distribution - Computed')

            n, bins, patches = ax[1].hist(distribution, alpha=.5, bins=bins, color='green', edgecolor='black', linewidth=1.2)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            ax[1].plot(bin_centers, n, color='r', linestyle='dashed', alpha=0.5)
            ax[1].set_xlabel('Value')
            ax[1].set_ylabel('Frequency')
            ax[1].set_title(f'{dist_name} Distribution - Numpy')           

class DiscreteRV(RV):
    def __init__(self, seed=42):
        '''
        set seed for reproducibility
        '''
        super().__init__(seed)

    def binomial(self, p=0.5, size=1):
        '''
        Method - Inverse Transform
        p: probability of success

        return: random variable array with 'size' number of samples
        '''
        assert 0 < p < 1, 'p must be between 0 and 1'
        self.p = p
        self.size = size

        return (np.random.rand(size) <= p).astype(int)
    
    def geometric(self, p=0.5, size=1):
        '''
        Method - Inverse Transform
        p: probability of success

        return: random variable array with 'size' number of samples
        '''
        assert 0 < p < 1, 'p must be between 0 and 1'

        self.p = p
        self.size = size

        u = np.random.rand(size)
        numer = np.log(1-u)
        denom = np.log(1-p)
        rv = np.ceil(numer/denom).astype(int)
        return rv
    
    def poisson(self, lamb=1, size=1):
        '''
        Method - Inverse Transform
        lam: rate

        return: random variable array with 'size' number of samples
        '''
        self.lamb = lamb
        self.size = size

        u = np.random.rand(size)
        pois = lambda x, l: (np.exp(-l)*l**x)/np.math.factorial(x)
        l = 3
        rv = []
        for num in u:
            x = 0
            cum_p = 0
            while True:
                cum_p += pois(x, lamb)
                if num <= cum_p:
                    rv.append(x)
                    break
                x += 1
        return np.array(rv)
    
    def discrete_rv(self, probabilities, values, size=1):
        '''
        Method - Inverse Transform
        probabilities: list of probabilities
        values: list of values

        return: random variable array with 'size' number of samples
        '''
        self.probabilities = probabilities
        self.values = values
        self.size = size

        u = np.random.rand(size)
        probs = np.cumsum(probabilities)
        rv = []
        for num in u:
            rv.append(values[np.searchsorted(probs, num)])
        rv = np.array(rv)
        return rv
    

class ContinuousRV(RV):
    def __init__(self, seed=42):
        '''
        set seed for reproducibility
        '''
        super().__init__(seed)

    def uniform(self, a=0, b=1, size=1):
        '''
        Method - Inverse Transform
        a: lower bound
        b: upper bound

        return: random variable array with 'size' number of samples
        '''
        assert a < b, 'a must be less than b'

        self.a = a
        self.b = b
        self.size = size

        u = np.random.rand(size)
        u = a + (b-a)*u
        return u

    def exponential(self, lamb=1, size=1):
        '''
        Method - Inverse Transform
        lamb: rate

        return: random variable array with 'size' number of samples
        '''
        self.lamb = lamb
        self.size = size

        u = np.random.rand(size)
        u = -np.log(u)
        u = u/lamb
        return u
    
    def weibull(self, lamb=1, b=1, size=1):
        '''
        Method - Inverse Transform
        lamb: scale
        b: shape

        return: random variable array with 'size' number of samples
        '''
        self.lamb = lamb
        self.b = b
        self.size = size

        u = np.random.rand(size)
        u = -np.log(u)
        u = (1/lamb)*u**(1/b)
        return u
    
    def normal(self, mu=0, sigma=1, size=1):
        '''
        Method - Box-Muller
        mu: mean
        sigma: standard deviation

        return: random variable array with 'size' number of samples
        '''
        self.mu = mu
        self.sigma = sigma
        self.size = size

        u1 = np.random.rand(size)
        u2 = np.random.rand(size)
        bm = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
        nor = mu + sigma*bm
        return nor
            
    def triangular(self, a=0, b=1, c=0.5, size=1):
        '''
        Method - Inverse Transform
        a: lower bound
        b: upper bound
        c: peak (must be between a and b)

        return: random variable array with 'size' number of samples
        '''
        assert a < b, 'a must be less than b'
        assert a <= c <= b, 'c must be between a and b'

        self.a = a
        self.b = b
        self.c = c
        self.size = size

        u = np.random.rand(size)
        u = np.where(u < (c-a)/(b-a), a + np.sqrt(u*(b-a)*(c-a)), b - np.sqrt((1-u)*(b-a)*(b-c))) 
        return u
    
    def chi_square(self, n=1, size=1):
        '''
        Method - Convolution (Sum of Squares of n Standard Normal Random Variables)
        n: degrees of freedom

        return: random variable array with 'size' number of samples
        '''
        self.n = n
        self.size = size

        chi = np.empty((n, size))
        for i in range(n):
            u = self.normal(mu=0, sigma=1, size=size)
            chi[i] = u**2
        chi = np.sum(chi, axis=0)
        return chi
