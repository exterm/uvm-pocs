import random
import scipy

# For γ = 5/2, generate n = 1000 sets each of N = 10, 102, 103, 104, 105, and 106 samples, using Pk = ck^(−5/2) with k = 1, 2, 3,...
c = 1 / scipy.special.zeta(5 / 2)
dist = lambda k: c * k**(-5 / 2)


def sample_from(distribution):
    """
    Generate a random number from a given distribution.
    """
    rand = random.uniform(0, 1)
    print(f"rand: {rand}")

    k = 1
    cumulated_probability = distribution(k)

    while cumulated_probability < rand and k <= 10**6:
        k += 1
        cumulated_probability += distribution(k)

    return k


for i in range(100):
    print(sample_from(dist))
