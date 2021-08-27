#!/usr/bin/env python3


# -------------------------------------------------------
# Create a bar plot with multiple datasets next to each
# other
# -------------------------------------------------------


from matplotlib import pyplot as plt
import numpy as np

np.random.seed(12345)

n_bins = 10
x = np.random.uniform(2, 3, size=(1000, 3))

fig, ((ax0, ax1)) = plt.subplots(nrows=2, ncols=1)

colors = ["red", "tan", "lime"]
ax0.hist(x, n_bins, density=True, histtype="bar", color=colors, label=colors)
ax0.legend(prop={"size": 10})
ax0.set_title("bars with legend")

# Make a multiple-histogram of data-sets with different length.
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
ax1.hist(x_multi, n_bins, histtype="bar")
ax1.set_title("different sample sizes")

fig.tight_layout()
plt.show()
