import random
import scipy
import numpy as np

# For γ = 5/2, generate n = 1000 sets each of N = 10, 10^2, 10^3, 10^4, 10^5, and 10^6 samples,
# using Pk = ck^(−5/2) with k = 1, 2, 3,...

gamma = 5 / 2
n = 1000
Ns = [10, 10**2, 10**3, 10**4, 10**5, 10**6]

c = 1 / scipy.special.zeta(gamma)
dist = lambda k: c * k**(-gamma)


def sample_from(distribution):
    """
    Generate a random number from a given discrete distribution.
    """
    rand = random.uniform(0, 1)

    k = 1
    cumulated_probability = distribution(k)

    while cumulated_probability < rand and k <= 10**6:
        k += 1
        cumulated_probability += distribution(k)

    return k


# print([sample_from(dist) for i in range(100)])

# a)
# For each value of sample size N, sequentially create n sets of N samples. For
# each set, determine and record the maximum value of the set’s N samples.


def max_set(N):
    """
    Return the maximum value of a set of N samples.
    """
    max_value = 0
    for i in range(N):
        max_value = max(max_value, sample_from(dist))
    return max_value


def max_sets(N, n):
    """
    Return the maximum values of n sets of N samples.
    """
    max_values = [max_set(N) for i in range(n)]
    return max_values


for N in Ns[:-1]:
    max_sets(N, n)
    print(f"{N} => {0}")

# b)
# Now find the average maximum value ⟨ik_{max,i}⟩ for each N
