#-------Libraries
import numpy as np
import matplotlib.pyplot as plt 
#--------------------------------

# this function finds the best homography using RANSAC
def compute_homography_ransac(src_points, dst_points, min_inliers, max_error):
    assert len(src_points) == len(dst_points), "Number of source and destination points must be equal"

    # Convert the points to homogeneous coordinates
    src_points = np.array(src_points)
    dst_points = np.array(dst_points)
    src_homogeneous = np.hstack((src_points, np.ones((len(src_points), 1))))
    dst_homogeneous = np.hstack((dst_points, np.ones((len(dst_points), 1))))

    num_points = len(src_points)
    best_num_inliers = 0
    best_homography = None

    for _ in range(1000):  # Number of RANSAC iterations
        # Randomly select minimal sample points
        indices = np.random.choice(num_points, 10, replace=False)
        src_sample = src_homogeneous[indices]
        dst_sample = dst_homogeneous[indices]

        # Compute the homography from the minimal sample
        homography = compute_homography(src_sample, dst_sample)

        # Compute the reprojection error for all points
        projected_points = np.dot(src_homogeneous, homography.T)
        projected_points /= projected_points[:, 2][:, np.newaxis]
        errors = np.linalg.norm(projected_points - dst_homogeneous, axis=1)

        # Count the inliers within the threshold
        num_inliers = np.sum(errors < max_error)

        # Update the best homography if the current one has more inliers
        if num_inliers >= min_inliers and num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_homography = homography
            global inliers_final
            inliers_final = indices

    return best_homography
#-----------------------------------------------------------------------

# this function finds homography assuming no outliers
def compute_homography(src_points, dst_points):
    num_points = src_points.shape[0]

    A = np.zeros((2 * num_points, 9))
    for i in range(num_points):
        x, y = src_points[i][:2]
        u, v = dst_points[i][:2]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]

    U, Sigma, V = np.linalg.svd(A)
    homography = V[-1].reshape(3, 3)

    return homography
#------------------------------------------------------------------

# Source points
src_points = [[1.90659, 2.51737],
              [2.20896, 1.1542],
              [2.37878, 2.15422],
              [1.98784, 1.44557],
              [2.83467, 3.41243],
              [9.12775, 8.60163],
              [4.31247, 5.57856],
              [6.50957, 5.65667],
              [3.20486, 2.67803],
              [6.60663, 3.80709],
              [8.40191, 3.41115],
              [2.41345, 5.71343],
              [1.04413, 5.29942],
              [3.68784, 3.54342],
              [1.41243, 2.6001]]
#--------------------------------------------------------------

# Destination points
dst_points = [[5.0513, 1.14083],
              [1.61414, 0.92223],
              [1.95854, 1.05193],
              [1.62637, 0.93347],
              [2.4199, 1.22036],
              [5.58934, 3.60356],
              [3.18642, 1.48918],
              [3.42369, 1.54875],
              [3.65167, 3.73654],
              [3.09629, 1.41874],
              [5.55153, 1.73183],
              [2.94418, 1.43583],
              [6.8175, 0.01906],
              [2.62637, 1.28191],
              [1.78841, 1.0149]]
#------------------------------------------------------------

# this is where the Compute the homography using RANSAC is called
homography = compute_homography_ransac(src_points, dst_points, min_inliers=10, max_error=0.005)

# Print the homography matrix
print("Best Homography matrix:")
print(homography)

src_points = np.array(src_points); dst_points = np.array(dst_points)

# original points
plt.scatter(src_points[:,0],dst_points[:,1],color='b',label='Source Points')
plt.scatter(dst_points[:,0],dst_points[:,1],color='r',label='Destination Points')
plt.legend(); plt.show() 

# Connecting inliers with lines
plt.scatter(src_points[:, 0], src_points[:, 1], marker='x', label='Outliers')
plt.scatter(dst_points[:, 0], dst_points[:, 1], marker='x')
plt.scatter(src_points[inliers_final, 0], src_points[inliers_final, 1], marker='o', label='Inliers')
plt.scatter(dst_points[inliers_final, 0], dst_points[inliers_final, 1], marker='o')
for i in inliers_final:
    plt.plot([src_points[i, 0], dst_points[i, 0]], [src_points[i, 1], dst_points[i, 1]], '-k')
plt.show()

