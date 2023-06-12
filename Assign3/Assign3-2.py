#------Libraries---------------------
import pygame
import numpy as np
from pygame.locals import *
from math import *
import matplotlib.pyplot as plt
import random
#--------------------------------------------------

# PyGame and its related parameteres are defined here
pygame.init()
dis_w = 1800; dis_h = 1600
gameDisplay = pygame.display.set_mode((dis_w,dis_h))
pygame.display.set_caption('Robot Position')
black = (0,0,0); white = (255,255,255); yellow=(234, 221, 202)
blue=(0,0,255); red1=(255,0,0); red2 = (180, 50, 50) ; green=(0,255,0)
clock = pygame.time.Clock()
crashed = False
#----------------------------------------------------------

# global variables are defined here
measured_positions_x=[]; measured_positions_y=[]
predicted_positions_x=[]; predicted_positions_y=[]
cov_width=[]; cov_hight=[]
M=np.array([[10],[10]])
points=[]; points.append([400,400]); points_measure=[]
points_gt=[]; points_gt.append([400,400])
points_measure.append([400,400])
position=np.array([[0],[0],[0]])
measured_position=position
p_0=np.array([[0,0,0],[0,0,0],[0,0,0]])
position_new_true = np.array([[0],[0],[0]])
particles_positions=np.array([[0],[0],[0]])
x_change = 0; t=1; iteration1=1
#-----------------------------------------------------------------------

# this function is the motion model
def motion_model(position,particles):
    global position_new_true
    A=np.array([[1,0,0],[0,1,0],[0,0,1]])
    x=[]; y=[]; r=0.1; l=0.3; dt=1/8;  u_r=u_l=1
    if (dist([position[0,0],position[1,0]],M)<10):
        u_r=1
        u_l=0
    if (dist([position[0,0],position[1,0]],M)>11):
        u_r=0
        u_l=1
    B=np.array([[r*dt*cos(position[2,0]),0],[r*dt*sin(position[2,0]),0],[0,dt*r/l]])
    B_t = np.array([[r*dt*cos(position_new_true[2,0]),0],[r*dt*sin(position_new_true[2,0]),0],[0,dt*r/l]])
    u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
    position_new=np.matmul(A,position)+np.matmul(B,u) + dt*np.array([[np.random.normal(0,0.01)],[np.random.normal(0,0.1)],[0]])
    position_new_true = np.matmul(A,position_new_true) + np.matmul(B_t,u)
    for i in range(0,len(particles)):
            if (dist([particles[i][0][0],particles[i][1][0]],M)<10):
                u_r=1
                u_l=0
            if (dist([particles[i][0][0],particles[i][1][0]],M)>11):
                u_r=0
                u_l=1
            u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
            temp=np.matmul(A,particles[i])+np.matmul(B,u) + dt*np.array([[np.random.normal(0,0.01)],[np.random.normal(0,0.1)],[0]])
            particles[i]=temp
            x.append(temp[0][0])
            y.append(temp[1][0])
    return position_new,particles,x,y
#-------------------------------------------------------------------------------

# this function calculates the sensor's measurment
def measurement():
    global measured_position
    C=np.array([[1,0,0],[0,2,0],[0,0,1]])
    R=np.array([[np.random.normal(0,0.05)],[np.random.normal(0,0.075)],[0]])
    Z = np.matmul(C,position_new_true) + R
    measured_position = Z
#---------------------------------------------------------------------------

# this function generates particles
def generate_particles():
    num_particles=100
    particles=[]
    particles_x=np.random.normal(position[0][0],0.001,num_particles)
    particles_y=np.random.normal(position[1][0],0.001,num_particles)
    particles_theta=np.random.normal(position[1][0],0,num_particles)
    for i in range(0,num_particles):
        particles.append(np.array([[particles_x[i]],[particles_y[i]],[particles_theta[i]]]))  
    return particles
#-------------------------------------------------------------------------

# this function computs the like1ihood of the measurement
def likelihood(x,y):
    pdf = (1.0/(2*np.pi*0.005*0.075))*np.exp(-(((x - measured_position[0][0])**2/(2*0.005*0.005))+((y - measured_position[1][0])**2/(2*0.075*0.075)))) + 0.0001    
    return pdf
#--------------------------------------------------------------------------

# this function computes the weights of eachh particle
def determine_weights(particles):
        global position_new_true
        PDF=np.zeros((1,len(particles)))
        for i in range(0,len(particles)):
            PDF[0,i]=likelihood(particles[i][0][0],particles[i][1][0])
        PDF=PDF/np.sum(PDF)
        x_n = 0; y_n = 0

        for i in range(0, len(particles)):	
            x_n += PDF[0,i] * particles[i][0][0]
            y_n += PDF[0,i] * particles[i][1][0]

        pose=np.array([[x_n],[y_n],[0]])
        return pose,PDF    
#---------------------------------------------------------------------

# this function carries out the resampling process
def resample(particles, PDF):
    num_particles=len(particles)
    new_particles = []
    index = int(random.random() * num_particles)
    t1 = 0.0
       
    for i in range(0,num_particles):
        PDF[0,i]=likelihood(particles[i][0][0],particles[i][1][0])
        
    PDF=PDF/np.sum(PDF)
    max_PDF = PDF.max()

    for i in range(num_particles):
        t1 += random.random()*2.0*max_PDF
        while t1 > PDF[0][index]:
            t1 -= PDF[0][index]
            index = (index + 1) % num_particles
        new_particles.append(np.array([[particles[index][0][0]],[particles[index][1][0]],[particles[index][2][0]]]))
       
    return new_particles    
#------------------------------------------------------------------

# this funcion calculates covariance elements
def determine_covariance(x,y):
    x=np.array(x); y=np.array(y)
    return np.mean(x),np.std(x),np.mean(y),np.std(y)
#------------------------------------------------------------------------

# this is the main body of the code where all funtions are called   
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
         
    if iteration1==1:
        particles=generate_particles()
        iteration1=0

    position,particles,x,y=motion_model(position,particles)
    mean_x,std_x,mean_y,std_y=determine_covariance(x,y)
     
    if(t%8==0):
        measurement()
        particles_positions,PDF=determine_weights(particles)
        particles=resample(particles, PDF) 
        
    gameDisplay.fill(white)
    for p in particles:
        pygame.draw.rect(gameDisplay,red1,(p[0][0]*1000+400,(p[1][0]/2)*1000+400,2,2))        
    surface = pygame.Surface((320, 240))
    pygame.draw.polygon(gameDisplay, blue,
                        [[particles_positions[0,0]*1000+400,particles_positions[1,0]/2*1000+400],[particles_positions[0,0]*1000+390,particles_positions[1,0]/2*1000+390] ,
                        [particles_positions[0,0]*1000+400,particles_positions[1,0]/2*1000+410]])

    size = (particles_positions[0,0]*1000+400-(std_x*2000)/2, particles_positions[1,0]/2*1000+400-(2000*std_y)/2, std_x*2000, 2000*std_y)
    pygame.draw.ellipse(gameDisplay, red1, size,1)
    points.append([particles_positions[0,0]*1000+400,particles_positions[1,0]/2*1000+400])
    points_gt.append([position_new_true[0,0]*1000+400,position_new_true[1,0]*1000+400])
    points_measure.append([measured_position[0,0]*1000+400,(measured_position[1,0]/2)*1000+400])
    pygame.draw.lines(gameDisplay,blue,False,points,5)
    pygame.draw.lines(gameDisplay,green,False,points_gt,5)
    pygame.draw.lines(gameDisplay,red1,False,points_measure,5)
    pygame.draw.rect(gameDisplay,yellow,(particles_positions[0,0]*1000+400,(particles_positions[1,0]/2)*1000+400,10,10))
    pygame.draw.rect(gameDisplay,red1,(measured_position[0,0]*1000+400,(measured_position[1,0]/2)*1000+400,10,10))
    pygame.draw.rect(gameDisplay,green,(position_new_true[0,0]*1000+400,(position_new_true[1,0])*1000+400,10,10))
    measured_positions_x.append(measured_position[0,0])
    measured_positions_y.append(measured_position[1,0])
    predicted_positions_x.append(position[0,0])
    predicted_positions_y.append(position[1,0])
    cov_hight.append(p_0[0,0])
    cov_width.append(p_0[1,1])
    pygame.display.update()
    clock.tick(8)
    t+=1
#-----------------------------------------------------------------------------

# this section plots all delivarables
plt.plot(measured_positions_x,label='Measured X')
plt.plot(measured_positions_y,label='Measured Y')
plt.xlabel("time step"); plt.ylabel("X or Y"); plt.legend(); plt.show()

plt.plot(predicted_positions_x,label='Predicted X')
plt.plot(predicted_positions_y,label='Predicted Y')
plt.xlabel("time step"); plt.ylabel("X or Y"); plt.legend(); plt.show()

pygame.quit()
quit()
#----------------------------------------------------------------------
