"""
Script to compute the killend branches by continuation of the conefield
computed by 'cKAM_CF_2r.py' 
[Implementation of converse KAM result (MacKay'18) to an integrable
magnetic field example: axisymmetrical magnetic field perturbed with
2 helical modes (Kallinikos'14).]     ~~ Conefield approach (Martinez '25)~~

 The script continues 4 initial condition (BLUE - boundary points)
 integrating the initial cones found on them by 'cKAM_CF_2r.py' to
 construct the ker(s^\pm) (i.e. killend branches) using a Euler integration. 
 The branches are continued with the aim to envelope the magnetic island merger
 and other structures not transverse to a raidal direction field xi.
 The script recomputates the cone on each continuation point/step.

 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 Special modules: "cf_tools.py" ,  "cf_comp.py"
 Inputs:  'ini.txt' - Parameter file
          'cf_{orbits}_{tf}_{ep1}_{m1}_{n1}_{ep2}_{m2}_{n2}.txt'
           from 'cKAM_CF_2r.py'

 Outputs  'ke_brch0_%s.txt' 
 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 Author: David Martinez del Rio (U.Warwick)

 based on the orignal codes by:
          Nikos Kallinikos (U.Warwick, U. W.Macedonia, Int Hellenic U.)
          David Martinez del Rio (U.Warwick)
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import simplejson
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp

from cf_tools import *  # Data format manipulation, and graphic - subroutines/definitions
from cf_comp import *   # Conefield computation and tensor (symbolic & numerical) subroutines/funtions


## SET RUN PARAMETERS ##
data = read_data()

tf = int(data[0]) #
Pts= int(data[1])

## IMPORTS DATA ##
e1 = data[2]
m1 = data[3]
n1 = data[4]
w1 = data[5]
w2 = data[6]
R0 = data[7]
e2 = data[8]
m2 = data[9]
n2 = data[10]

pla = int(data[11]) # Plotting region selection
# [0] - whole plane 
# [1] - upper plane
# [2] - positive quadrant
points = Pts**2 #
if pla == 1:
    points = int (points/2)
    Pts = int(Pts/2)

#String formating for labels - 
ep1 = ep2str(e1)
ep2 = ep2str(e2)

## OPENS CF FILE ##
file = open('cf_%s_%s_%s_%s_%s_%s_%s_%s.txt' % (points, tf,ep1, int(m1),int(n1),ep2, int(m2),int(n2)), 'r')
result = simplejson.load(file)
file.close()



# Data read / variable allocation - - - - - - - - - - - - - - - - - - - - -  
x1p = []  # Red points
x2p = []
x1u = []  # Blue points
x2u = []

x1 = []  # y-R0 value in (y,z)
x2 = []  # z    value in (y,z)
q  = []  # te/tf : relative time of detection
mp = []  # upper slope value
mm = []  # Lower slope bound
thc = [] # Angle value of the (x10,y10) cartesian point
rc = []  # Radial value of the (x10,y10) cartesian point
nf = []  # Boundary points index
btr=0    # Blue-tp-Red boundary identifier variable
tf = int(tf/2) # Half of integration time

for i in range(len(result)) :
    x10 = result[i][0] - R0 
    x20 = result[i][1]
    ie  = result[i][2]
    if abs(x10) + abs(x20)> 0.05:
        te  = result[i][3]
    else:
        te = tf
    mm0  = result[i][5]
    mp0  = result[i][6]
    if ie == 1 and abs(x10) + abs(x20)> 0.01:
        x1p.append(x10)   # Red points
        x2p.append(x20)   #
        btr +=1
        if btr ==1 and x20 >0 and i >0:
            nf.append(i-1) # Outer (island) point
    else :
        x1u.append(x10)   # Blue points
        x2u.append(x20)   #
        if btr>0:
            nf.append(i)   # Outer (island) point
            btr = 0
    x1.append(x10)
    x2.append(x20)
    q.append(te/tf)
    r0 = np.sqrt(x10**2 + x20**2)
    rc.append(r0)
    thc.append(np.arctan2(x20,x10))
    mm.append(mm0)   # Last slopes computed
    mp.append(mp0)   #


# Island - BOUNDARY
# Simple boundary finding algorithm | Boundary: Blue points next To a Red point = = =
bnd = []
for i in range(len(result)) :
    b = 1
    if i+Pts < len(result):
        if q[i]== 1 and q[i+Pts] < 1 and x2[i] < 0.6: # Blue to Red | Horizontal
            b = 0
    if i+1 < len(result):
        if q[i]== 1 and q[i+1] < 1 and x2[i] < 0.6:   # Blue to red | vertical
            b = 0
    if i-Pts > -1:
        if q[i-Pts] < 1 and q[i] == 1: # Red to Blue | Horizontal
            b = 0
    if i > 0:
        if q[i-1] < 1 and q[i] == 1 and x2[i]> 0:   # Red to blue | vertical
            b = 0 
    if b < 1:
        bnd.append(i+b)


# Identification of the points for continuation
ke_s = []
ke_s0 = []
orb0 = []
orb1 = []
orb2 = []
orb3 = []
orb4 = []


# First and second point of the right branches | - - - o - - - P1 r r P2 - - 
k=0
for j in range(len(bnd)):  # Identifying points in the boundary
    nn = bnd[j]            # that cross x1 > 0 (z = 0) - Right semi axis
    if x1[nn] > 0.01 and x2[nn] == 0:
        # Seeds identification                       Integrate
        #               y0     z0     m_-    m_+     c-Clkwise
        ke_s0.append([x1[nn] ,x2[nn],mm[nn],mp[nn], k,  1])
        k += 1

# print(ke_s0)

ke_s.append([ke_s0[0][0],  ke_s0[0][1],  ke_s0[0][2],  ke_s0[0][3],  0, 1])
ke_s.append([ke_s0[-1][0], ke_s0[-1][1], ke_s0[-1][2], ke_s0[-1][3], 1, 1])
k=2
    

# Third point for the inner left branch | r r P3 - - o - - -  
y3 = -0.7
for j in range(len(bnd)):  # Identifying points in the boundary
    nn = bnd[j]            # that cross x1 < 0 (z = 0) - Left semi axis
    if x1[nn] < -0.01 and x2[nn] == 0:
        if x1[nn] > y3:
            y3,z3,ma,mb = x1[nn],x2[nn],mm[nn],mp[nn]  # Choose the closest
        #               m_-   m_+   #  Clkwise         # to the origin
ke_s.append([y3,z3, ma,   mb,   k,  -1])
k += 1


# Fourth point for the exterior left branch | P4 \ r \ r \ \  o - - - 
dmin = 1
yb = -0.42
zb = 0.6
for j in range(len(bnd)):  # Identifying the closest point in the boundary 
    nn = bnd[j]            # to the eliminated point.
    #dist = np.abs(x1[nn]*m4 - x2[nn])
    dist = (yb - x1[nn])**2 + (zb - x2[nn])**2
    if dist < dmin :  # Minimal distance 
        dmin = dist
        ya = x1[nn]
        za = x2[nn]
        mmin_a  = mm[nn]
        mplus_a = mp[nn]
        
ke_s.append([ya,za, mmin_a,   mplus_a,   k,  -1])

# Fifth point for the exterior left branch | P5 \ r \ r \ \  o - - -
# Same point as 4th but opposite direction for continuation
ke_s.append([ya,za, mmin_a,   mplus_a,   k+1,  1])

# print('Seed points')
# print(ke_s[0])
# print(ke_s[1])
# print(ke_s[2])
# print(ke_s[3])
# print(ke_s[4])


## Initial values and parameters ~ ~ ~
B0 = [1,0,0,0,1,0,0,0,1]  
par = [e1,m1,n1,e2,m2,n2,R0,w1,w2]
delta = 0.005   # Continuation step <<<<
Nk = [int(0.7/delta),int(0.9/delta),int(0.9/delta), #<<<< Continuation number of steps
      int(1.2/delta),int(0.9/delta)]                #     of each branch
              
# tf = int(tf/2) # Total integration interval: [-tf, tf]
t_int_b = [0, -tf]   # Backward integration time 
t_int_f = [0, tf]    # Forward integration time


# Computation of the four continuation branches
# for k in range(0,1): # For testing
for k in range(0,5): # 5 Branches
    ie = 0.0  # initial status (Blue point) 
    y0 = ke_s[k][0]
    z0 = ke_s[k][1]
    y__0 = y0  # Previous iterate 
    z__0 = z0  # ini values
    mmin =  ke_s[k][2]
    mplus = ke_s[k][3]
    
    for i in range(Nk[k]):
        if ke_s[k][4] == 0:    # P1
            m = mmin           # Lower bound R  | Slope: /
            m2 = mplus         # For continuation if (y0,z0) interior point 
            orb0.append([y0,z0,mmin,mplus,ie])
        if ke_s[k][4] == 1:    # P2 
            m  = mplus         # Upper bound L  | Slope: \
            m2 = mmin          # For continuation if (y0,z0) exterior point
            orb1.append([y0,z0,mmin,mplus,ie])
        if ke_s[k][4] == 2:    # P3
            m  = mplus         # Upper bound L  | Slope: \
            m2 = mmin          # For continuation if (y0,z0) interior point
            orb2.append([y0,z0,mmin,mplus,ie])
        if ke_s[k][4] == 3:    # P4
            m  = mmin          # Lower bound R  | Slope: /
            m2 = mplus         # For continuation if (y0,z0) exterior point
            orb3.append([y0,z0,mmin,mplus,ie])
        if ke_s[k][4] == 4:    # P5
            m  = mplus         # Lower bound L  | Slope: \
            m2 = mmin          # For continuation if (y0,z0) exterior point 
            orb4.append([y0,z0,mmin,mplus,ie])
            
        # Euler step : used slope varies depending of ke_s[k][5] (1 for c-Clockwise and -1 clockwise)
        dy = delta / np.sqrt((1+m**2))   # 
        if   ke_s[k][5] * ( y0*m - z0 ) > 0:
            y0 = y0 +  dy
            z0 = z0 +  m * dy
        else:
            y0 = y0 -    dy
            z0 = z0 - m* dy 

        if z0 < 0: break  # Break for the 5th Branch
        
        
        ψ0 = ψf2(0,y0,z0,R0)
        vth0 = thf2(0,y0,z0,R0)  
        ph0 = 0
        Z0 =  [ψ0, vth0, ph0,*B0,*par]
         
        SOLDBf = solve_ivp(system3, t_int_f, Z0, method='RK45', rtol=1e-9, atol=1e-11)
        SOLDBb = solve_ivp(system3, t_int_b, Z0, method='RK45', rtol=1e-9, atol=1e-11)
        
        # Select smallest lenght of integration to evaluate the conefield cKAM criterium
        t_test = len(SOLDBf.t)
        if len(SOLDBf.t) > len(SOLDBb.t):
            t_test = len(SOLDBb.t)

        DBf = np.transpose(SOLDBf.y)
        DBb = np.transpose(SOLDBb.y)
        
        for j in range(0,t_test):
        
            sp,sm,mplus, mmin,mp_0, mm_0 =cf_computation(ψ0, vth0, ph0,DBf[j],DBb[j],*par)

            if sp < sm:
                te = SOLDBf.t[j]
                ie = 1.0
                break
            else :
                te = tf
                ie = 0.0
        dmin = 1
        if ie == 1.0:
            if ke_s[k][4] ==0:                     # Save the first eliminated point
                orb0.append([y0,z0,mmin,mplus,ie]) # that appears in the continuation
            if ke_s[k][4] == 1:                    # i.e. last point of
                orb1.append([y0,z0,mmin,mplus,ie]) # a first continuation
            if ke_s[k][4] == 2:                    # 
                orb2.append([y0,z0,mmin,mplus,ie]) # 
            if ke_s[k][4] == 3:                    # 
                orb3.append([y0,z0,mmin,mplus,ie]) #
            if ke_s[k][4] == 4:                    # 
                orb4.append([y0,z0,mmin,mplus,ie]) # 
            
            # Continuation point by one iteration with opposite (outward) slope
            if  ke_s[k][5] * ( y0*m - z0 ) > 0:
                y0 = y__0 +  dy
                z0 = z__0 +  m2 * dy
            else:
                y0 = y__0 -    dy
                z0 = z__0 - m2* dy
            ie = 2.0   # To mark the transitions
            
            
        y__0 = y0 # Previous iterate values
        z__0 = z0
    print('Branch '+ str(k+1)+ ' DONE')


np.savetxt('ke_brch0_%s.txt' % (2*tf), orb0)
np.savetxt('ke_brch1_%s.txt' % (2*tf), orb1)
np.savetxt('ke_brch2_%s.txt' % (2*tf), orb2)
np.savetxt('ke_brch3_%s.txt' % (2*tf), orb3)
np.savetxt('ke_brch4_%s.txt' % (2*tf), orb4)


