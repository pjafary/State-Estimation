import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as patches

# defining x
rho=1
theta=np.deg2rad(0.5) 
x=(rho, theta) 

# creating covariance of x
sigma_x=np.zeros((2,2))
sigma_x[0,0]=0.01
sigma_x[1,1]=1

# defining jacobian
jacob=np.array([[np.cos(theta),-rho*np.sin(theta)],[np.sin(theta), rho*np.cos(theta)]])

# finding covariance of y from covariance of x
sigma_y=np.zeros((2,2))
sigma_y=np.matmul(np.matmul(jacob,sigma_x),np.transpose(jacob))

# defining normal distribution y
y=(rho*np.cos(theta), rho*np.sin(theta)) 
sample = np.random.multivariate_normal(y, sigma_y, size=1000)

# calculating eigen values and vectors
eigenvalues, eigenvectors = np.linalg.eig(sigma_y)
angle=np.arctan2(eigenvectors[0][1],eigenvectors[0][0])

# plots
fig, ax = plt.subplots()
ax.scatter(sample[:, 0], sample[:, 1], marker='.', color='red', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(0.5,1.5)
ax.set_ylim(-2.5,2.5)
ax.set_title('Transformed Points and Ellipse')
ax.grid(True)


s=np.sqrt(5.991)
width = 2 * np.sqrt(eigenvalues[0])*s
height = 2 * np.sqrt(eigenvalues[1])*s

angle=np.arctan2(eigenvectors[0][1],eigenvectors[0][0])

ellipse = Ellipse((rho*np.cos(theta), rho*np.sin(theta)), width, height, color='green', angle=angle, alpha=0.3)
ax.add_patch(ellipse)

plt.show()