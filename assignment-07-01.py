import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

# For γ = 5/2, generate n = 1000 sets each of N = 10, 10^2, 10^3, 10^4, 10^5, and 10^6 samples,
# using Pk = ck^(−5/2) with k = 1, 2, 3,...

# In z = (1 − F(z))^(−2/3) we replace F(z) with our random number x and round
# the value of z to finally get an estimate of k.

part = 'b'  # execute part 'a' or 'b'. Added switch to save some execution time.
gamma = 5 / 2
n = 1000

# Save lots of time when testing by shortening this list
Ns = [10, 10**2, 10**3, 10**4, 10**5, 10**6]  #[:-3]

# explicitly initialize the random number generator to make results reproducible
rng = np.random.RandomState(0)


def ks(count):
    rands = rng.random(count)
    zs = (1 - rands)**(-1 / (gamma - 1))
    ks = zs.round()
    return ks


def kmaxs(N, n):
    """
    Return the maximum values of n sets of N samples.
    """
    max_values = [np.amax(ks(N)) for i in range(n)]
    return max_values


def kmaxs_sets(Ns, n):
    sets = {}
    for N in Ns:
        print(f"Generating data for N = {N}")
        sets[N] = kmaxs(N, n)
    print("Done.")
    return sets


sets = kmaxs_sets(Ns, n)

# a)
# For each value of sample size N, sequentially create n sets of N samples. For
# each set, determine and record the maximum value of the set’s N samples.

if part == 'a':

    # plot the n values of k_(max,i) as a function of i.

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.3, left=0.1, wspace=0.3)

    i = 0
    j = 0
    for N, set_k in sets.items():
        print(f"Plotting data for N = {N}")
        axes[i, j].plot(range(1, len(set_k) + 1), set_k)
        axes[i, j].set_xlabel('Set #')
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
else:
    averages = {}
    for N, set_k in sets.items():
        averages[N] = np.mean(set_k)
        print(
            f"Average of sets of 10^{int(math.log10(N))} samples: {averages[N]}"
        )

# Plot ⟨kmax⟩ as a function of N on double logarithmic axes, and calculate the scaling using
# least squares. Report error estimates.

    averages_df = pd.DataFrame.from_dict(averages,
                                         columns=['Average'],
                                         orient='index')

    # Calculate the scaling via linear regression.
    def regression_confidence(percent, sample_size, regression_result):
        # we only have a sample, know nothing about the population, so we should use
        #  the t-statistic
        tinv = lambda p, df: abs(stats.t.ppf(p / 2, df))
        ts = tinv(1 - (percent / 100.), sample_size - 2)
        print(f"slope ({percent}%): {regression_result.slope:.6f} +/-"
              f"{ts*regression_result.stderr:.6f}")
        print(f"intercept ({percent}%): {regression_result.intercept:.6f}"
              f" +/- {ts*regression_result.intercept_stderr:.6f}")

    def regress_ll(x, y):
        result = stats.linregress(np.log10(x), np.log10(y))
        regression_confidence(95, len(x), result)
        return result

    regression = regress_ll(averages_df.index, averages_df['Average'])

    # Plot.

    print("Plotting averages.")
    plt.plot(averages_df, 'o', label='Averages')

    print("Plotting regression line.")
    intercept = 10**regression.intercept * 10
    plt.plot(averages_df.index,
             intercept * np.power(averages_df.index, regression.slope),
             color='g',
             label=r"Regression $\gamma=%.2f$" % -regression.slope)
    plt.legend()
    plt.loglog(10)
    plt.xlabel('Sample size')
    plt.ylabel('$\\langle k_{\\rm max} \\rangle$')
    plt.savefig('output/assignment-07-01b.png', dpi=600)
    plt.show()
