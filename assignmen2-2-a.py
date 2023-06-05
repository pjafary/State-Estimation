#------Libraries-------------------------------------------
from hashlib import new
from re import U
import pygame
import numpy as np
from pygame.locals import *
from math import *
from matplotlib.patches import Ellipse
import multiprocessing
import matplotlib.pyplot as plt
#----------------------------------------------------------

# PyGame and its related parameteres are defined here
pygame.init()
dis_w = 1800; dis_h = 1600
gameDisplay = pygame.display.set_mode((dis_w,dis_h))
pygame.display.set_caption('Robot Position')
black = (0,0,0); white = (255,255,255)
white=(255,255,255); blue=(0,0,255); red1=(255,0,0); red2 = (180, 50, 50); green=(0,255,0)
clock = pygame.time.Clock()
crashed = False
#-------------------------------------------------------------

# global parameters are defined here
measured_positions_x=[]; measured_positions_y=[]
predicted_positions_x=[]; predicted_positions_y=[]
cov_width=[]; cov_hight=[]
points=[]; points.append([400,400])
points_measure=[]; points_measure.append([400,400])
points_gt=[]; points_gt.append([400,400])
position=np.array([[0],[0],[0]])
measured_position=position
p_0=np.array([[0,0,0],[0,0,0],[0,0,0]])
true_position = np.array([[0],[0],[0]])
A=np.array([[1,0,0],[0,1,0],[0,0,1]])
r=0.1; l=0.3; dt=1/8
Q=np.array([[0.01,0,0],[0,0.1,0],[0,0,0]])*dt
H=np.array([[1,0,0],[0,2,0],[0,0,1]])
R=np.array([[0.05,0,0],[0,0.075,0],[0,0,0]])
M=np.array([[10],[10]])
x_change = 0
#-------------------------------------------------------------

# this function carries out the position estimation 
def position_estimation(position):
    global true_position
    print(position)
    print(dist([position[0,0]*10,position[1,0]*10],M))
    if (dist([position[0,0],position[1,0]],M)<10):
        u_r=1; u_l=0

    if (dist([position[0,0],position[1,0]],M)>11):
        u_r=0; u_l=1

    B=np.array([[r*dt*cos(position[2,0]),0],[r*dt*sin(position[2,0]),0],[0,dt*r/l]])
    G_true = np.array([[r*dt*cos(true_position[2,0]),0],[r*dt*sin(true_position[2,0]),0],[0,dt*r/l]])
    u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
    w_psi = [np.random.normal(0,0.01)]; w_omega = [np.random.normal(0,0.1)]
    position_new=np.matmul(A,position)+np.matmul(B,u) + dt*np.array([w_psi,w_omega,[0]])
    true_position = np.matmul(A,true_position) + np.matmul(G_true,u)
     
    return position_new
#--------------------------------------------------------------------------------

# this function updates the covariance matrix at each time step 
def covariance():
    global p_0 
    temp=np.matmul(A,p_0)
    p_new=np.matmul(temp,A.transpose())+Q
    p_0=p_new
#---------------------------------------------------------------------------------
 
# this function computes the sensor's measurment  
def measurement():
    global measured_position
    global position
    w_z=np.array([[np.random.normal(0,0.05)],[np.random.normal(0,0.075)],[0]])
    Z = np.matmul(H,true_position) + w_z
    measured_position = Z
#--------------------------------------------------------------------------------

# this function computes Kalman gain at each time step   
def kalman():
    global p_0
    t1=np.matmul(p_0,H.transpose())
    t2=np.matmul(H,p_0)
    t3=np.matmul(t2,H.transpose())+R
    k=t1/t3
    k[np.isnan(k)] = 0
  
    return H,k
#--------------------------------------------------------------------------------

# this function takes previously calculated C, K and computes the position
def update(H,K):
    global p_0; global position
    t1=np.identity(3)-np.matmul(K,H)
    p_0=np.matmul(t1,p_0)
    t2=(measured_position - np.matmul(H,position))
    position = position + np.matmul(K,t2)
#--------------------------------------------------------------------------------

# this is the main loop in which all the functions are called
t=1   
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

    position=position_estimation(position)

    covariance()

    if(t%8==0):
        measurement()
        H_new,K_new=kalman()
        update(H_new,K_new)
        
    gameDisplay.fill(white)
    
    surface = pygame.Surface((320, 240))

    pygame.draw.polygon(gameDisplay, blue,
                        [[position[0,0]*1000+400,position[1,0]*1000+400],[position[0,0]*1000+390,position[1,0]*1000+390] ,
                        [position[0,0]*1000+400,position[1,0]*1000+410]])

    size = (position[0,0]*1000+400-(p_0[0,0]*2000)/2, position[1,0]*1000+400-(2000*p_0[1,1])/2, p_0[0,0]*2000, 2000*p_0[1,1])
    pygame.draw.ellipse(gameDisplay, red2, size,1)
    points.append([position[0,0]*1000+400,position[1,0]*1000+400])
    points_gt.append([true_position[0,0]*1000+400,true_position[1,0]*1000+400])
    points_measure.append([measured_position[0,0]*1000+400,(measured_position[1,0]/2)*1000+400])
    pygame.draw.lines(gameDisplay,blue,False,points,5)
    pygame.draw.lines(gameDisplay,green,False,points_gt,5)
    pygame.draw.lines(gameDisplay,red1,False,points_measure,5)
    
    if(t%8==0):
        pygame.draw.rect(gameDisplay,red1,(measured_position[0,0]*1000+400,(measured_position[1,0]/2)*1000+400,10,10))
        pygame.draw.rect(gameDisplay,green,(true_position[0,0]*1000+400,(true_position[1,0]/2)*1000+400,10,10))

    pygame.draw.rect(gameDisplay,red1,(measured_position[0,0]*1000+400,(measured_position[1,0]/2)*1000+400,10,10))
    pygame.draw.rect(gameDisplay,green,(true_position[0,0]*1000+400,(true_position[1,0])*1000+400,10,10))
    measured_positions_x.append(measured_position[0,0])
    measured_positions_y.append(measured_position[1,0])
    predicted_positions_x.append(position[0,0])
    predicted_positions_y.append(position[1,0])
    cov_hight.append(p_0[0,0])
    cov_width.append(p_0[1,1])
    pygame.display.update()
    clock.tick(8)
    t+=1
#---------------------------------------------------------------------------------

# this section plots all deliverables
plt.plot(predicted_positions_x,label='Predicted X')
plt.plot(predicted_positions_y,label='Predicted Y')
plt.xlabel("time step"); plt.ylabel("X or Y"); plt.legend(); plt.show()

plt.plot(measured_positions_x,label='Measured X')
plt.plot(measured_positions_y,label='Measured Y')
plt.xlabel("time step"); plt.ylabel("X or Y"); plt.legend(); plt.show()

plt.plot(cov_hight,label='Covariance of X')
plt.plot(cov_width,label='Covariance of Y')
plt.xlabel("time step"); plt.ylabel("Covariance of X or Y"); plt.legend(); plt.show()

pygame.quit()
quit()
#---------------------------------------------------------------------------------
