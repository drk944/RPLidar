import numpy as np
import matplotlib.pyplot as plt
import math

# Load data npy file
points1 = np.load("/home/laser14/Documents/Personal/RPLidar/ICP/test_data/bd1.npy", allow_pickle=True)[0]
points2 = np.load("/home/laser14/Documents/Personal/RPLidar/ICP/test_data/bd2.npy", allow_pickle=True)[0]

Q = np.column_stack((points1[0].T,points1[1].T))
P = np.column_stack((points2[0].T,points2[1].T))

# Find correspondances between points in P and Q
def compute_correspondence(P, Q):
    correspondences = []
    total_dist = 0
    for i in range(P.shape[0]):
        min_dist = np.inf
        for j in range(Q.shape[0]):
            dist = np.linalg.norm(P[i] - Q[j])
            if dist < min_dist:
                min_dist = dist
                min_correspondence_idx = j
        correspondences.append(min_correspondence_idx)
        total_dist += min_dist
        
    return correspondences, total_dist

def plot_correspondence(P, Q, correspondences):
    for i in range(P.shape[0]):
        plt.plot([P[i,0], Q[correspondences[i],0]], [P[i,1], Q[correspondences[i],1]], 'r-')
    plt.plot(Q[:,0], Q[:,1], 'o', label='Q: Scan 1', markersize=1)
    plt.plot(P[:,0], P[:,1], 'o', label='P: Scan 2', markersize=1)
    plt.legend()
    plt.axis('equal')

# Data needs to be centered for the ICP algorithm to work
def center_data(data):
    center = np.array([data.mean(axis=0)])
    return center, data - center

center_of_P, P_centered = center_data(P)

def compute_cross_covariance(P, Q, correspondences):
    cross_covariance = np.zeros((2,2))
    for i in range(P.shape[0]):
        cross_covariance += np.outer(P[i], Q[correspondences[i]])
    return cross_covariance

def SVD(Q, P, cross_covariance):
    U, S, V = np.linalg.svd(cross_covariance)
    R = np.dot(U, V.T) 
    center_of_Q = center_data(Q)[0]
    t = (center_of_Q.T - R.T@center_of_P.T).T[0]
    # R is transposed because indices are flipped col, row
    return R, t

def SVD_ICP(Q, P, num_iterations=10, plotting=False):
    loss = []
    center_of_P, P_centered = center_data(P)
    for i in range(num_iterations):
        correspondences, sum_error = compute_correspondence(P_centered, Q)
        if plotting and i % 10 == 0:
            plot_correspondence(P_centered, Q, correspondences)
            plt.title("Iteration: " + str(i))
            plt.pause(0.1)
            plt.clf()
        cross_covariance = compute_cross_covariance(P_centered, Q, correspondences)
        R, t = SVD(Q, P_centered, cross_covariance)
        P_centered = P_centered@R+t
        center_of_P, P_centered = center_data(P_centered)
        loss.append(sum_error)

    return R, t, loss

R, t, loss = SVD_ICP(Q, P, num_iterations=100, plotting=True)
print("Rotation matrix:\n", R)
print("Translation vector:\n", t)
# print("Rotation Matrix Error:", np.linalg.norm(R - true_rotation))
# t is multiplied by -1 because the translation direction is flipped
# print("Translation Vector Error:", np.linalg.norm(t*-1 - true_translation))

plt.plot(loss, label='Scan Difference')
plt.grid()
plt.title("Iterative Loss")
plt.xlabel("Iteration")
plt.ylabel("Abs Diff between P and Q")
plt.legend()
plt.show()