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
from sqlalchemy import false


def plot_points(points1, points2):
    plt.scatter(points1[:, 0], points1[:, 1], label="Scan 1", s=1)
    plt.xlim(-6000, 2000)
    plt.ylim(-4000, 6000)

    plt.scatter(points2[:, 0], points2[:, 1], c='r', label="Scan 2", s=1)
    plt.xlim(-6000, 2000)
    plt.ylim(-4000, 6000)


def center_data(data, exclude_indices=[]):
    reduced_data = np.delete(data, exclude_indices, axis=1)
    center = np.array([reduced_data.mean(axis=0)]).reshape(2, 1).T
    return center.T, data - center


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
points1 = np.load("ICP/scans.npy", allow_pickle=True)[0]
points2 = np.load("ICP/scans1.npy", allow_pickle=True)[0]
points1 = np.column_stack((points1[0].T, points1[1].T))
points2 = np.column_stack((points2[0].T, points2[1].T))
plt.figure(1)
plot_points(points1, points2)


center_of_P, P_centered = center_data(points1)
center_of_Q, Q_centered = center_data(points2)
plt.figure(2)
plot_points(P_centered, Q_centered)

correspondences = find_correspondences(P_centered, Q_centered)
draw_correspondeces(P_centered, Q_centered, correspondences)
plt.show(block=False)


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


cov, _ = compute_cross_covariance(P_centered, Q_centered, correspondences)
print(cov)


U, S, V_T = np.linalg.svd(cov)
print(S)
R_found = U.dot(V_T)
t_found = center_of_Q - R_found.dot(center_of_P)
print("R_found =\n", R_found)
print("t_found =\n", t_found)


def icp_svd(P, Q, iterations=10, kernel=lambda diff: 1.0):
    """Perform ICP using SVD."""
    center_of_Q, Q_centered = center_data(Q)
    norm_values = []
    P_values = [P.copy()]
    P_copy = P.copy()
    corresp_values = []
    exclude_indices = []
    for i in range(iterations):
        center_of_P, P_centered = center_data(
            P_copy, exclude_indices=exclude_indices)
        correspondences = find_correspondences(P_centered, Q_centered)
        corresp_values.append(correspondences)
        # norm_values.append(np.linalg.norm(P_centered - Q_centered))
        cov, exclude_indices = compute_cross_covariance(
            P_centered, Q_centered, correspondences, kernel)
        U, S, V_T = np.linalg.svd(cov)
        R = U.dot(V_T)
        t = center_of_Q - R.dot(center_of_P)
        P_copy = (R.dot(P_copy.T) + t).T
        P_values.append(P_copy)
    corresp_values.append(corresp_values[-1])
    return P_values, norm_values, corresp_values


P_values, norm_values, corresp_values = icp_svd(points1, points2)
plt.figure(3)
plot_points(P_values[-1], points2)
plt.show()
