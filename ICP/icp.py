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
    plt.scatter(points1[:,0], points1[:,1], label="Scan 1", s=2)
    plt.xlim(-6000, 1300)
    plt.ylim(-4000, 6000)

    plt.scatter(points2[:,0], points2[:,1], c='r', label="Scan 2", s=2)
    plt.xlim(-6000, 1300)
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
        x = [P[i, 0], Q[j, 0]]
        y = [P[i, 1], Q[j, 1]]
        if not label_added:
            plt.plot(x, y, color='grey', label='correpondences')
            label_added = True
        else:
            plt.plot(x, y, color='grey')
    plt.legend()


# fixme: format of scans is 2 arrays of x and y values, points 2 is longer than points 1
points1 = np.load("scans.npy", allow_pickle=True)[0]
points2 = np.load("scans1.npy", allow_pickle=True)[0]

points1 = np.column_stack((points1[0].T,points1[1].T))
points2 = np.column_stack((points2[0].T,points2[1].T))


correspondences = find_correspondences(points1, points2)
draw_correspondeces(points1, points2, correspondences)
plot_original(points1, points2)
plt.show()

def compute_cross_covariance(P, Q, correspondences, kernel=lambda diff: 1.0):
    cov = np.zeros((2, 2))
    exclude_indices = []
    for i, j in correspondences:
        p_point = P[[i], :]
        q_point = Q[[j], :]
        weight = kernel(p_point - q_point)
        if weight < 0.01:
            exclude_indices.append(i)
        cov += weight * q_point.dot(p_point.T)
    return cov, exclude_indices


cov, _ = compute_cross_covariance(points1, points2, correspondences)
print(cov)

U, S, V_T = np.linalg.svd(cov)
print(S)
R_found = U.dot(V_T)
t_found = center_of_Q - R_found.dot(center_of_P)
print("R_found =\n", R_found)
print("t_found =\n", t_found)