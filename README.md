#### Random Variate Generator Library

Steps to use rand_var_gen

1. create a new python environment using python=3.12.3
    - ex: conda create -n "name_of_env" python=3.12.3
2. Install dependencies from requirements.txt
    - pip install -r requirements.txt
3. Make rand_var_gen the current working directory
4. Open new python file or notebook
5. Import the random variate generator classes
    - from random_var import DiscreteRv, ContinuousRV
6. Instantiate discrete or continuous rv object, setting seed is optional
    - drv = DiscreteRV(seed=1234)
    - crv = ContinuousRV(seed=1234)
7. Based on the desired sample data, run one of the mehtods
    - poisson_data = drv.poisson(lamb=2, size=1000)
    - normal_data = crv.normal(mu=2, sigma=3, size=1000)
8. To plot distribution, use plot_hist
    - drv.plot_hist(drv, 'name of dist', bins=20)
    - to compare histogram to numpy, 'name of dist' must be one of the following based on which to compare to
        - Binomial
        - Geometric
        - Poisson
        - Discrete
        - Uniform
        - Exponential
        - Weibull
        - Normal
        - Triangular
        - Chi-Square
9. To run goodness of fit tests
    - import TestDiscreteRV and TestContinuousRV from test_rv
    - instantiate object
        - testdrv = TestDiscreteRV()
        - testcrv = TestContinuousRV()
    - Run one of the tests
        - testdrv.test_geometric()
    - test statistic and p-value will be returned
