#%%
from random_var import DiscreteRV, ContinuousRV
# %%
# to create discrete random variable object
# set seed for reproducibility
drv = DiscreteRV(seed=2134)
#%%
rv = drv.geometric(p=.3, size=10000)
rv[:20]
#%%
drv.plot_dist(rv, 'Geometric', bins=15, plot_comparison=False)

#%%
rv = drv.binomial(p=.3, size=10000)
drv.plot_dist(rv, 'Binomial', bins=2, plot_comparison=True)

# %%
rv = drv.poisson(lamb=2, size=10000)
drv.plot_dist(rv, 'Poisson', bins=10, plot_comparison=True)
# %%
rv = drv.discrete_rv(probabilities=[0.2, 0.5, 0.3], values=[1, 2, 3], size=10000)
drv.plot_dist(rv, 'Discrete', bins=3, plot_comparison=True)
# %%
# to create continuous random variable object
# set seed for reproducibility
crv = ContinuousRV(seed=2134)
# %%
rv = crv.uniform(a=0, b=5, size=10000)
crv.plot_dist(rv, 'Uniform', bins=10, plot_comparison=True)
# %%
rv = crv.exponential(lamb=1, size=10000)
crv.plot_dist(rv, 'Exponential', bins=10, plot_comparison=True)
# %%
rv = crv.weibull(lamb=1, b=2, size=10000)
crv.plot_dist(rv, 'Weibull', bins=10, plot_comparison=True)
# %%
rv = crv.normal(mu=10, sigma=3, size=10000)
crv.plot_dist(rv, 'Normal', bins=20, plot_comparison=True)
# %%
rv = crv.triangular(a=0, c=5, b=10, size=10000)
crv.plot_dist(rv, 'Triangular', bins=20, plot_comparison=True)
#%%
rv = crv.chi_square(n=20, size=10000)
crv.plot_dist(rv, 'Chi-Square', bins=20, plot_comparison=True)

#%%

# to run tests
from test_rv import TestDiscreteRV

tdrv = TestDiscreteRV()
print(tdrv.test_geometric(p=0.3, size=1000))
