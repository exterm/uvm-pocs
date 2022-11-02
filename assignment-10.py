import functools

import pandas as pd
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

# Code up the discrete HOT model in 2d. Let’s see if we find any of these super-duper power laws everyone keeps talking
#  about. We’ll follow the same approach as the N = L×L 2-d forest discussed in lectures.
# Main goal: extract yield curves as a function of the design D parameter as described below.
# Suggested simulations elements:
# - Take L = 32 as a start. Once your code is running, see if L = 64, 128, or more might be possible. (The original
#   sets of papers used all three of these values.) Use a value of L that’s sufficiently large to produce useful
#   statistics but not prohibitively time consuming for simulations.
# - Start with no trees.
# - Probability of a spark at the (i; j)th site: P(i; j) ∝ e^(−i/ℓ)e^(−j/ℓ) where (i; j) is the tree position with the
#   indices starting in the top left corner (i; j = 1 to L) (You will need to normalize this properly). The quantity ℓ
#   is the characteristic scale for this distribution. Try out ℓ = L/10.
# - Consider a design problem of D = 1, 2, L, and L^2. (If L and L^2 are too much, you can drop them. Perhaps sneak out
#   to D = 3.) Recall that the design problem is to test D randomly chosen placements of the next tree against the
#   spark distribution.
# - For each test tree, compute the average forest fire size over the full spark distribution: ∑(i;j)P (i; j)S(i; j);
#   where S(i; j) is the size of the forest component at (i; j). Select the tree location with the highest average
#   yield and plant a tree there.
# - Add trees until the 2-d forest is full, measuring average yield as a function of trees added.
# - Only trees within the cluster surrounding the ignited tree burn
#   (trees are connected through four nearest neighbors).
#
# (a) Plot the forest at (approximate) peak yield.
# (b) Plot the yield curves for each value of D, and identify (approximately) the peak yield and the density for which
#     peak yield occurs for each value of D.
# (c) Plot Zipf (or size) distributions of tree component sizes S at peak yield.
#     Note: You will have to rebuild forests and stop at the peak yield value of D to find these distributions. By
#     recording the sequence of optimal tree planting, this can be done without running the simulation again.
# (d) Extra level: Plot Zipf (or size) distributions for D = L2 for varying tree densities
#     \rho = 0:10; 0:20; : : : ; 0:90. This will be an effort to reproduce Fig. 3b in [2].
# Hint: Working on un-treed locations will make choosing the next location easier.

def simulate_tree_at_location(world: np.ndarray, tree_planting_location: np.ndarray):
    # plant tree
    world[tree_planting_location[0], tree_planting_location[1]] = True

    # compute the average forest fire size over the full spark distribution:
    #  Sum over all fields of lightning probability multiplied by component size
    component_sizes = calculate_component_size_map(world)

    L = world.shape[0]

    average_forest_fire_size = np.sum(lightning_probability_distribution(L) * component_sizes)

    # remove tree
    world[tree_planting_location[0], tree_planting_location[1]] = False

    return average_forest_fire_size

@functools.cache
def lightning_probability_distribution(L: int):
    # lightning probability: e^(−i/ℓ)e^(−j/ℓ), where ℓ = L/10
    l = L / 10
    lightning_probabilities = np.exp(-np.arange(L) / l) * np.exp(-np.arange(L) / l).reshape(-1, 1)
    lightning_probabilities /= lightning_probabilities.sum()
    return lightning_probabilities

def calculate_component_size_map(world: np.ndarray):
    labeled_components, _ = ndimage.label(world)
    # replace component labels with component size
    component_sizes = np.bincount(labeled_components.flatten())
    component_sizes[0] = 0
    component_sizes = component_sizes[labeled_components]
    return component_sizes

# Run HOT model for given D
# Return the yield curve and the forest at peak yield
def run_hot_model(L: int, D: int):
    print("running HOT model for L = %d and D = %d" % (L, D))

    # empty world
    world = np.zeros((L, L), dtype=bool)

    yields = []
    max_yield = 0
    max_yield_forest = None

    # add trees until forest is full
    # that means we have to add L^2 trees
    for _ in range(L * L):
        # choose D random empty locations for tree planting simulations
        empty_locations = np.array(np.where(world == 0)).T
        tree_planting_locations = empty_locations[
            np.random.choice(empty_locations.shape[0], min(D, len(empty_locations)), replace=False)
        ]

        average_forest_fire_sizes = [
            simulate_tree_at_location(world, location) for location in tree_planting_locations
        ]

        # plant tree
        tree_planting_location = tree_planting_locations[np.argmin(average_forest_fire_sizes)]
        world[tree_planting_location[0], tree_planting_location[1]] = True

        # the yield is the number of trees minus the average forest fire size
        yield_ = np.sum(world) - np.min(average_forest_fire_sizes)

        yields += [yield_]

        if yield_ > max_yield:
            max_yield = yield_
            max_yield_forest = world.copy()

    return yields, max_yield_forest

L = 10
Ds = [1, 2, L, L ** 2]
results = [run_hot_model(L, D) for D in Ds]

# (a) Plot the forest at (approximate) peak yield.
# (draw subplots for each D in two rows)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, (yields, max_yield_forest) in enumerate(results):
    axes[i // 2, i % 2].imshow(max_yield_forest)
    axes[i // 2, i % 2].set_title("D = %d" % Ds[i])
plt.draw()

# (b) Plot the yield curves for each value of D, and identify (approximately) the peak yield and the density for which
#     peak yield occurs for each value of D.
#     (draw into one plot, x: density, y: yield)
plt.figure()
for i, (yields, max_yield_forest) in enumerate(results):
    plt.plot(np.arange(len(yields)) / (L * L), yields, label="D = %d" % Ds[i])
plt.legend()
plt.xlabel("density")
plt.ylabel("yield")

# (c) Plot Zipf (or size) distributions of tree component sizes S at peak yield.
# (draw subplots for each D in two rows)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, (yields, max_yield_forest) in enumerate(results):
    labeled_components, _ = ndimage.label(max_yield_forest)
    component_sizes = np.bincount(labeled_components.flatten())[1:]
    axes[i // 2, i % 2].hist(component_sizes, bins=range(1, L*L))
    axes[i // 2, i % 2].set_title("D = %d" % Ds[i])
    axes[i // 2, i % 2].set_yscale("log")
    axes[i // 2, i % 2].set_xlabel("component size")
    axes[i // 2, i % 2].set_ylabel("count")

plt.draw()

plt.show()
