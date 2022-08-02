# Goal: Minimize the distances between point pairs
#       1. Compute Centre of Mass
#       2. Comput the rotation using SVD
# Estimate and recompute
#
# ICP Approaches:
# Point-to-plane*
# Least Squares*
# Projective ICP
# Robust Kernels to ignore outliers

# Steps
# 1. Make data centered by subtracting the mean
# 1-A. Doesn't really apply since the lidar is always at the origin
# 2. Find correspondences for each point in P
# 3. Perform a single iteration by computing the cross-covariance matrix and performing the SVD
# 4. Apply the found rotation to P
# 5. Repeat until correspondences don't change
# 6. Apply the found rotation to the mean vector of P and uncenter P with it

import numpy as np
import matplotlib.pyplot as plt


def plot_original(points1, points2):
    plt.scatter(points1[0], points1[1], label="Scan 1", s=1)
    plt.xlim(-6000, 1200)
    plt.ylim(-4000, 6000)

    plt.scatter(points2[0], points2[1], c='r', label="Scan 2", s=1)
    plt.xlim(-6000, 1200)
    plt.ylim(-4000, 6000)

# Find correspondances between points in P and Q


def find_correspondences(points1, points2):
    correspondences = []
    for i in range(len(points1)):
        p_point = points1[i]
        min_dist = np.inf
        for j in range(len(points2)):
            q_point = points2[j]
            dist = np.linalg.norm(p_point - q_point)
            if dist < min_dist:
                min_dist = dist
                chosen_idx = j
        correspondences.append((i, chosen_idx))
    return correspondences


def draw_correspondeces(P, Q, correspondences):
    label_added = False
    for i, j in correspondences:
        x = [P[0, i], Q[0, j]]
        y = [P[1, i], Q[1, j]]
        if not label_added:
            plt.plot(x, y, color='grey', label='correpondences')
            label_added = True
        else:
            plt.plot(x, y, color='grey')
    plt.legend()


# fixme: format of scans is 2 arrays of x and y values, points 2 is longer than points 1
points1 = np.load("scans.npy", allow_pickle=True)[0]
points2 = np.load("scans1.npy", allow_pickle=True)[0]
points2 = points2[0:471]

plot_original(points1, points2)

correspondences = find_correspondences(points1, points2)
draw_correspondeces(points1, points2, correspondences)
plt.show()
