import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt

# Load data npy file
# points1 = np.load("/home/laser14/Documents/Personal/RPLidar/ICP/test_data/bd1.npy", allow_pickle=True)[0]
# points2 = np.load("/home/laser14/Documents/Personal/RPLidar/ICP/test_data/bd2.npy", allow_pickle=True)[0]

def non_linear_ICP(scan1, scan2, plotting=False):

    Q = np.column_stack((scan1[0].T,scan1[1].T))
    P = np.column_stack((scan2[0].T,scan2[1].T))

    def plot_correspondence(P, Q, correspondences):
        for i in range(P.shape[0]):
            plt.plot([P[i,0], Q[correspondences[i],0]], [P[i,1], Q[correspondences[i],1]], 'r-')
        plt.plot(Q[:,0], Q[:,1], 'o', label='Q: Scan 1', markersize=1)
        plt.plot(P[:,0], P[:,1], 'o', label='P: Scan 2', markersize=1)
        plt.legend()
        plt.axis('equal')

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

    def dR(theta):
        return np.array([[-np.sin(theta), -np.cos(theta)], [np.cos(theta), -np.sin(theta)]])
    def R_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def jac(x, p_point):
        theta = x[2]
        J = np.zeros((2,3))
        J[0:2, 0:2] = np.eye(2)
        J[0, 2] = -p_point[0]*np.sin(theta) - p_point[1]*np.cos(theta)
        J[1, 2] = p_point[0]*np.cos(theta) - p_point[1]*np.sin(theta)
        return J

    def error(x, p_point, q_point):
        theta = x[2]
        rotation = R_matrix(theta)
        translation = x[0:2]
        prediction = rotation.dot(p_point.reshape(2,1)).T[0] + translation
        return prediction.T - q_point


    def iterate_icp_least_squares(inputs, p_points, q_points, correspondences):
        H = np.zeros((3,3))
        g = np.zeros((3,1))
        chi = 0
        for i in range(len(correspondences)):
            p_point = p_points[i]
            q_point = q_points[correspondences[i]]
            e = error(inputs, p_point, q_point).T
            J = jac(inputs, p_point)
            H += J.T.dot(J)
            g += np.reshape((J.T.dot(e)), (3,1))
            chi += e.T.dot(e)
        return H, g, chi

    def icp_least_squares(P, Q, max_iterations, plotting=False):
        x = np.zeros((3)) # x, y, theta
        chi_values = []
        x_values = [x.copy()] # adding first value to array
        P_values = [P.copy()] # adding first value to array
        P_copy = P.copy() # copy of P to be used in the loop
        correp_values = []
        num_iterations = 0
        while True:
            rot = R_matrix(x[2])
            t = x[0:2]
            correspondences, sum_error = compute_correspondence(P_copy, Q)
            correp_values.append(correspondences)
            H, g, chi = iterate_icp_least_squares(x, P, Q, correspondences)
            dx = np.linalg.lstsq(H, -g, rcond=None)[0]
            x = np.reshape(x + dx.T, (3))
            x[2] = math.atan2(np.sin(x[2]), np.cos(x[2])) # Normalize theta
            
            rot = R_matrix(x[2])
            t = x[0:2]
            # Fix me, needto add t to all points in P
            P_copy = (rot.dot(P.T.copy())).T + t # Update P with new rotation and translation
            
            # P_copy[:,0] += t[0]
            # P_copy[:,1] += t[1]
            chi_values.append(chi) # Append chi value to track error
            
            x_values.append(x.copy())
            P_values.append(P_copy)
            if plotting and num_iterations % 5 == 0:
                plot_correspondence(P_copy, Q, correspondences)
                plt.title("Iteration: " + str(i))
                plt.pause(0.1)
                plt.clf()

            if num_iterations > 2:
                a = chi_values[-1]
                b = chi_values[-2]
                diff = abs(a-b)/((a+b)/2)
                # print(num_iterations, ":", diff)
                if diff < 0.05:
                    break
            if num_iterations > max_iterations:
                break
            
            num_iterations += 1

            # plt.plot(Q[:,0], Q[:,1], 'o', label='Q: Scan 1', markersize=8)
            # plt.plot(P_copy[:,0], P_copy[:,1], 'o', label='P: Scan 2')
            # plt.xlim(-10, 10)
            # plt.ylim(-10, 10)
            # plt.show()

        correp_values.append(correp_values[-1]) # don't understand this line
        return x_values, P_values[-1], chi_values, correp_values

    x_values, P_values, chi_values, correp_values = icp_least_squares(P, Q, max_iterations=100, plotting=plotting)
    # print(x_values[-1])
    # print()
    # print(chi_values)

    if plotting:
        plt.plot(Q[:,0], Q[:,1], 'o', label='Q: Scan 1', markersize=1)
        plt.plot(P_values[:,0], P_values[:,1], 'o', label='P: Scan 2', markersize=1)
        # plt.xlim(-10, 10)
        # plt.ylim(-10, 10)
        plt.pause(0.5)
        # plt.show()
    return x_values[-1], chi_values, len(x_values)

if __name__ == "__main__":
    # Load data npy file
    points1 = np.load("/home/laser14/Documents/Personal/RPLidar/ICP/test_data/bd1.npy", allow_pickle=True)[0]
    points2 = np.load("/home/laser14/Documents/Personal/RPLidar/ICP/test_data/bd2.npy", allow_pickle=True)[0]
    t, chi, num_evals = non_linear_ICP(points1, points2, plotting=True)
    print(t)
    # print(chi)
    print(num_evals)