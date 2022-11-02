import functools
import argparse
import time
import cProfile

from scipy import ndimage
from scipy.stats import rankdata
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
# (d) Extra level: Plot Zipf (or size) distributions for D = L^2 for varying tree densities
#     \rho = 0:10; 0:20; : : : ; 0:90. This will be an effort to reproduce Fig. 3b in [2].
# Hint: Working on un-treed locations will make choosing the next location easier.

def simulate_tree_at_location(world: np.ndarray, tree_planting_location: np.ndarray, component_size_map: np.ndarray):
    thisworld = world.copy()
    # plant tree
    thisworld[tree_planting_location[0], tree_planting_location[1]] = True

    # compute the average forest fire size over the full spark distribution:
    #  Sum over all fields of lightning probability multiplied by component size
    if isolated(thisworld, tree_planting_location):
        # shortcut: if the tree is isolated, we know no existing components will be changed.
        component_sizes = component_size_map.copy()
        component_sizes[tree_planting_location[0], tree_planting_location[1]] = 1
    else:
        component_sizes = calculate_component_size_map(thisworld)

    L = thisworld.shape[0]
    average_forest_fire_size = np.sum(lightning_probability_distribution(L) * component_sizes)

    return average_forest_fire_size

def isolated(world: np.ndarray, tree_location: np.ndarray) -> bool:
    x, y = tree_location
    # check if the tree is isolated
    # - define fields outside of the world as empty
    # - use von Neumann neighborhood
    # - make sure we don't go out of bounds
    return (world[x - 1, y] == 0 if x > 0 else True) and \
        (world[x + 1, y] == 0 if x < world.shape[0] - 1 else True) and \
        (world[x, y - 1] == 0 if y > 0 else True) and \
        (world[x, y + 1] == 0 if y < world.shape[1] - 1 else True)

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
    history = []
    max_yield = 0
    max_yield_forest = None

    # add trees until forest is full
    # that means we have to add L^2 trees
    for i in range(L * L):
        # print progress
        if i % 100 == 0:
            print("  %d/%d" % (i, L * L))
        # choose D random empty locations for tree planting simulations
        empty_locations = np.array(np.where(world == 0)).T
        tree_planting_locations = empty_locations[
            np.random.choice(empty_locations.shape[0], min(D, len(empty_locations)), replace=False)
        ]

        component_size_map = calculate_component_size_map(world)

        average_forest_fire_sizes = [
            simulate_tree_at_location(world, location, component_size_map) for location in tree_planting_locations
        ]

        # plant tree
        tree_planting_location = tree_planting_locations[np.argmin(average_forest_fire_sizes)]
        world[tree_planting_location[0], tree_planting_location[1]] = True

        # the yield is the number of trees minus the average forest fire size
        yield_ = np.sum(world) - np.min(average_forest_fire_sizes)

        yields += [yield_]
        history += [world.copy()]

        if yield_ > max_yield:
            max_yield = yield_
            max_yield_forest = world.copy()

    return yields, max_yield_forest, history

argparser = argparse.ArgumentParser()
argparser.add_argument("L", type=int, default=10)
argparser.add_argument("--no-plot", action="store_true")
argparser.add_argument("--profile-to", type=str)

args = argparser.parse_args()

L = args.L
Ds = [1, 2, L, L ** 2]

# measure execution time
if args.profile_to:
    profile = cProfile.Profile()
    profile.enable()
start = time.time()
results = [run_hot_model(L, D) for D in Ds]
end = time.time()
if args.profile_to:
    profile.disable()
    profile.dump_stats(args.profile_to)
print("execution time: %.2f seconds" % (end - start))

if not args.no_plot:
    print("plotting results...")

    # (a) Plot the forest at (approximate) peak yield.
    # (draw subplots for each D in two rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, (yields, max_yield_forest, _) in enumerate(results):
        axes[i // 2, i % 2].imshow(max_yield_forest)
        axes[i // 2, i % 2].set_title("D = %d" % Ds[i])
        # remove ticks
        axes[i // 2, i % 2].set_xticks([])
        axes[i // 2, i % 2].set_yticks([])

    fig.tight_layout()
    plt.savefig(f"output/10_3_a-L{L}.png", dpi=600)

    # (b) Plot the yield curves for each value of D, and identify (approximately) the peak yield and the density for
    #     which peak yield occurs for each value of D.
    #     (draw into one plot, x: density, y: yield)
    plt.figure()
    for i, (yields, max_yield_forest, _) in enumerate(results):
        plt.plot(np.arange(len(yields)) / (L * L), yields, label="D = %d" % Ds[i])
    plt.legend()
    plt.xlabel("density")
    plt.ylabel("yield")
    plt.savefig(f"output/10_3_b-L{L}.png", dpi=600)

    # (c) Plot Zipf distributions of tree component sizes S at peak yield.
    # (draw subplots for each D in two rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, (yields, max_yield_forest, _) in enumerate(results):
        labeled_components, _ = ndimage.label(max_yield_forest)
        component_sizes = np.bincount(labeled_components.flatten())[1:]
        axes[i // 2, i % 2].scatter(component_sizes, rankdata(-component_sizes))
        axes[i // 2, i % 2].set_xscale("log")
        axes[i // 2, i % 2].set_yscale("log")
        axes[i // 2, i % 2].set_title("D = %d" % Ds[i])
        axes[i // 2, i % 2].set_yscale("log")
        axes[i // 2, i % 2].set_xlabel("component size")
        axes[i // 2, i % 2].set_ylabel("rank")

    # increase spacing between subplots
    fig.tight_layout()
    plt.savefig(f"output/10_3_c-L{L}.png", dpi=600)

    # (d) Extra level: Plot Zipf distributions for D = L^2 for varying tree densities
    L_squared_history = results[-1][-1]

    # filter history for desired densities
    densities = np.arange(0.1, 1, 0.1)
    history = [L_squared_history[int(density * L * L)] for density in densities]

    plt.figure()
    for i, forest in enumerate(history):
        labeled_components, _ = ndimage.label(forest)
        component_sizes = np.bincount(labeled_components.flatten())[1:]
        plt.scatter(component_sizes, rankdata(-component_sizes), label="density = %.2f" % densities[i])
    plt.legend()
    plt.xlabel("component size")
    plt.ylabel("component size rank")
    plt.loglog(10)
    plt.tight_layout()
    plt.savefig(f"output/10_3_d-L{L}.png", dpi=600)
    plt.draw()

    print("done")

    plt.show()
