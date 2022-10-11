import numpy as np

# For γ = 5/2, generate n = 1000 sets each of N = 10, 10^2, 10^3, 10^4, 10^5, and 10^6 samples,
# using Pk = ck^(−5/2) with k = 1, 2, 3,...

# In z = (1 − F(z))^(−2/3) we replace F(z) with our random number x and round
# the value of z to finally get an estimate of k.

gamma = 5 / 2
n = 1000
Ns = [10, 10**2, 10**3, 10**4, 10**5, 10**6]


def ks(bunchsize):
    rands = np.random.rand(bunchsize)
    zs = (1 - rands)**(-(gamma - 1))
    ks = zs.round()
    return ks


def kmax(N):
    """
    Return the maximum value of a set of N samples.
    """
    return np.amax(ks(N))


def kmaxs(N, n):
    """
    Return the maximum values of n sets of N samples.
    """
    max_values = [kmax(N) for i in range(n)]
    return max_values


# a)
# For each value of sample size N, sequentially create n sets of N samples. For
# each set, determine and record the maximum value of the set’s N samples.


def kmaxs_sets(Ns, n):
    sets = {}
    for N in Ns:
        print(f"Generating data for N = {N}")
        sets[N] = kmaxs(N, n)
    print("Done.")
    return sets


sets = kmaxs_sets(Ns, n)

# plot the n values of k_(max,i) as a function of i.
import matplotlib.pyplot as plt
import math

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.subplots_adjust(hspace=0.3, left=0.1)

i = 0
j = 0
for N, set_k in sets.items():
    print(f"Plotting data for N = {N}")
    axes[i, j].plot(range(1, len(set_k) + 1), set_k)
    axes[i, j].set_xlabel('Number of set')
    axes[i, j].set_ylabel('Maximum value')
    axes[i, j].set_title('Maxima of sets of 10^{} samples'.format(
        int(math.log10(N))))

    if j == 2:
        i += 1
        j = 0
    else:
        j += 1

print("Saving output.")
plt.savefig('output/assignment-07-01a.png', dpi=600)
print("Showing output.")
plt.show()

# b)
# Now find the average maximum value ⟨ik_{max,i}⟩ for each N
