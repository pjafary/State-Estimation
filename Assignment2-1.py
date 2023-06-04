#------Libraries---------------------
from cProfile import label
from cmath import atan
from hashlib import new
from re import U
import pygame
import numpy as np
import matplotlib.pyplot as plt 
from pygame.locals import *
#-----------------------------------------------------

# PyGame and its related parameteres are defined here
pygame.init()
dis_w = 1800; dis_h = 1600
gameDisplay = pygame.display.set_mode((dis_w,dis_h))
pygame.display.set_caption('Robot Position')
black = (0,0,0); white = (255,255,255)
blue=(0,0,255); red1=(255,0,0); red2 = (180, 50, 50) ; green=(0,255,0)
clock = pygame.time.Clock()
crashed = False
#-----------------------------------------------------

# global parameters are defined here
points=[]; points.append([0,0])
measured_points=[]; measured_points.append([0,0])
points_gt=[]; points_gt.append([0,0])
measured_positions_x=[]; measured_positions_y=[]
predicted_positions_x=[]; predicted_positions_y=[]
cov_width=[]; cov_hight=[]
position=np.array([[0],[0]])
point_prev=[[0],[0]]
x_change = 0
measured_position=position
p_0=np.array([[0,0],[0,0]])
teta=-1; r=0.1; dt=1/8
A=np.array([[1,0],[0,1]]); 
B=np.array([[r/2*dt,r/2*dt],[r/2*dt,r/2*dt]])
u=np.array([[1],[1]])
R=np.array([[0.05,0],[0,0.075]]) #covariance of wx,wy
C=np.array([[1,0],[0,2]])
Q=np.array([[0.1,0],[0,0.15]])*dt
t=1   
#-----------------------------------------------------

# this function carries out the position estimation 
def position_estimation(position):
    global position_new_true
    w_x=np.random.normal(0,0.1); w_y=np.random.normal(0,0.15) # x,y noise
    position_new_true = np.matmul(A,position) + np.matmul(B,u)
    position_new = position_new_true + np.array([[w_x],[w_y]])*dt
    return position_new
#---------------------------------------------------------------

# this function computes the sensor's measurment  
def measurement():
    global measured_position
    r_x = np.random.normal(0,0.05); r_y = np.random.normal(0,0.075)
    Z=np.matmul(C,position_new_true) + np.array([r_x,r_y])
    measured_position=Z
#----------------------------------------------------------------

# this function updates the covariance matrix at each time step 
def covariance():
    global p_0
    t1=np.matmul(A,p_0)
    p_new=np.matmul(t1,A.transpose()) + Q
    p_0=p_new
#---------------------------------------------------------------
    
# this function computes Kalman gain at each time step   
def kalman_gain():
    t1=np.matmul(p_0,C.transpose())
    t2 = np.matmul(C,p_0)
    t3=np.matmul(t2,C.transpose()) + R
    k=t1/t3
    k[np.isnan(k)] = 0
    return C,k
#-----------------------------------------------------------------

# this function takes previously calculated C, K and computes the position
def update(C,K):
    global p_0, position
    t1= np.identity(2) - np.matmul(C,K)
    p_0=np.matmul(t1,p_0)
    t2= measured_position - np.matmul(C,position)
    position = position + np.matmul(K,t2)
    print("pos!", position)
#-----------------------------------------------------------------

# this is the main loop in which all the functions are called
while not crashed:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

    position=position_estimation(position)

    covariance()

    if(t%8==0):
        measurement()
        C,K_new=kalman_gain()
        update(C,K_new)

    gameDisplay.fill(white)
    
    size = (position[0,0]*1000+50-(p_0[0,0]*2000)/2, position[1,0]*1000+50-(2000*p_0[1,1])/2, p_0[0,0]*2000, 2000*p_0[1,1])

    pygame.draw.ellipse(gameDisplay, red2, size,1)  
    
    pygame.draw.polygon(gameDisplay, blue,
                        [[position[0,0]*1000+50,position[1,0]*1000+50],[position[0,0]*1000+40,position[1,0]*1000+35] ,
                        [position[0,0]*1000+40,position[1,0]*1000+65]])
  
    points.append([position[0,0]*1000+50,position[1,0]*1000+50])
    points_gt.append([position_new_true[0,0]*1000+50,position_new_true[1,0]*1000+50])
    measured_points.append([measured_position[0,0]*1000+50,(measured_position[1,0]/2)*1000+50])
    pygame.draw.lines(gameDisplay,blue,False,points,5)
    pygame.draw.lines(gameDisplay,green,False,points_gt,5)
    pygame.draw.lines(gameDisplay,red1,False,measured_points,5)

    if(t%8==0):
        pygame.draw.rect(gameDisplay,red1,(measured_position[0,0]*1000+50,(measured_position[1,0]/2)*1000+50,10,10))
        pygame.draw.rect(gameDisplay,green,(position_new_true[0,0]*1000+50,(position_new_true[1,0]/2)*1000+50,10,10))

    pygame.draw.rect(gameDisplay,red1,(measured_position[0,0]*1000+50,(measured_position[1,0]/2)*1000+50,10,10))
    pygame.draw.rect(gameDisplay,green,(position_new_true[0,0]*1000+50,(position_new_true[1,0])*1000+50,10,10))   
    measured_positions_x.append(measured_position[0,0])
    measured_positions_y.append(measured_position[1,0])
    predicted_positions_x.append(position[0,0])
    predicted_positions_y.append(position[1,0])
    cov_hight.append(p_0[0,0])
    cov_width.append(p_0[1,1])
    
    pygame.display.update()
    clock.tick(8) 
        
    t+=1
#----------------------------------------------------------------------

# this section plots all delivarables
plt.plot(measured_positions_x,label='Measured X')
plt.plot(measured_positions_y,label='Measured Y')
plt.xlabel("time step"); plt.ylabel("X or Y"); plt.legend(); plt.show()

plt.plot(predicted_positions_x,label='Predicted X')
plt.plot(predicted_positions_y,label='Predicted Y')
plt.xlabel("time step"); plt.ylabel("X or Y"); plt.legend(); plt.show()

plt.plot(cov_hight,label='Covariance of X')
plt.plot(cov_width,label='Covariance of Y')
plt.xlabel("time step"); plt.ylabel("Covariance of X or Y"); plt.legend(); plt.show()

pygame.quit()
quit()
#----------------------------------------------------------------------