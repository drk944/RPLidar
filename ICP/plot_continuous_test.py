import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
import Non_linear_ICP
import ICP_SVD
import time

prev_scan = 0
# Load data npy file
scans = np.load("ICP/test_data/test_track_5.npy", allow_pickle=True)
for scan_iter in range(len(scans)):
    if scan_iter == 0:
        prev_scan = scans[0]
        continue
    curr_scan = scans[scan_iter]
    start_time = time.time()
    opt_t, chi, num_icp_iterations = Non_linear_ICP.non_linear_ICP(prev_scan, curr_scan, plotting=False)    
    nls_time = time.time() - start_time
    start_time = time.time()
    svd_t, loss, num_SVD_iterations = ICP_SVD.ICP_SVD(prev_scan, curr_scan, plotting=False)
    svd_time = time.time() - start_time
    print("Non-linear ICP time: ", nls_time, "at", num_icp_iterations, "iterations")
    print("SVD ICP time: ", svd_time, "at", num_SVD_iterations, "iterations")
    t = opt_t
    # t = svd_t
    print("Num Iterations:", num_icp_iterations)
    # print("Num Iterations:", num_SVD_iterations)
    # print(chi)
    
    # t = svd_t
    # plt.plot(loss)
    # plt.show()
    print(t)
    print("Translation:", np.sqrt(t[0]**2+t[1]**2))
    print()
    # break

    
    prev_scan = scans[scan_iter]