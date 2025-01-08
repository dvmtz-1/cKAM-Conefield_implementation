"""
 Script to use a spline interpolation on the killends branches
 computed by 'ke_branch_comp.py' on conefiled data from 'cKAM_CF_2r.py'
 and plot the area enclosed by the branches.
 
 [Implementation of converse KAM result (MacKay'18) to an integrable
 magnetic field example: axisymmetrical magnetic field perturbed with
 2 helical modes (Kallinikos'14).]  ~~ Conefield approach (Martinez'25) ~~
 
 # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  
 Inputs:  'ini.txt' - Parameter file
 
          'ke_{orbits}_{tf}_{ep1}_{m1}_{n1}_{ep2}_{m2}_{n2}.txt'
                    - Conefiled data from 'cKAM_CF_2r.py'
             
          'ke_brch[N]_[tf].txt' : N=0,...,4
                    - Killend branches from 'ke_branch_comp.py'

 # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 Author: David Martinez del Rio (U.Warwick)
"""

import matplotlib.pyplot as plt
import numpy as np
import simplejson
from scipy import interpolate
from cf_tools import *  # Data format manipulation, and graphic - subroutines/definitions





## Data reading and variable allocation 
data = read_data()
tf = int(data[0])

ke1 = read_data4('ke_brch0_'+ str(tf))   # L: 0---P1 r r
ke2 = read_data4('ke_brch1_'+ str(tf))   # L: 0---   r r P2
ke3 = read_data4('ke_brch2_'+ str(tf))   # R: r r P3 - - 0 - - -  
ke4 = read_data4('ke_brch3_'+ str(tf))   # R: P4 \ r \ r \ \  0 - - - 
ke5 = read_data4('ke_brch4_'+ str(tf))   # L: P5 \ r \ r \ \  0 - - -



## Monotonicity test in theta: returns idex of first point where the monotonicity
## breaks and remainder of untested points in the array
def test_mono(num,ke,sgn):
    ket = ke.transpose()
    th_points = np.arctan2(sgn*ket[1],ket[0])
    th0 = th_points[0]
    for i in range(1,len(ket[0])):
        if th_points[i] > th0:
            th0 = th_points[i]
        else:
            break
    return i, len(ket[1])-i-1 

## Remove points in the branch that are not monotonic in theta
def monotonize(ke0,brnum,sgn):
    dc , rem = test_mono(brnum,ke0,sgn)
    it = 0
    while rem != 0 and it < 1000:
        ke0 = np.delete(ke0, (dc), axis=0)
        dc , rem = test_mono(brnum,ke0,sgn)
        it += 1
    print(' test', brnum ,'= ',test_mono(brnum,ke0,sgn), '|   Pt_removed =',it)
    return ke0

 
## Edition of killend branches to ensure the monotonizity on angle variable
## to ensure spline interpolation tools converge
ke1 = monotonize(ke1,'1',1)
ke2 = monotonize(ke2,'2',1)
ke3 = monotonize(ke3,'3',-1)
ke4 = monotonize(ke4,'4',-1)
ke5 = monotonize(ke5,'5',1)


# Find points of minimum distance between two branches
def branches_min(k1,k2):
    dmin = 0.1
    d = 0.2
    for i in range(len(k1)):
        for j in range(len(k2)):
            d = (k1[i][0] - k2[j][0])**2 + (k1[i][1] - k2[j][1])**2
            if d < dmin:
                i0 = i
                j0 = j
                dmin = d
    return i0 ,j0, dmin

# Testing for crossings ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
i1 ,j3 ,dmin1 = branches_min(ke1,ke3) # Minimum between branch 1 and 3
# print('i1,j3, dmin = ',branches_min(ke1,ke3))
i2 ,j4 ,dmin2 = branches_min(ke2,ke4) # minimum between branch 2 and 4
# print('i2,j4, dmin = ',branches_min(ke2,ke4))



## Spline interpolation of the branches # - - - - - - - - SPLINES %
## Branch 1
ke1_exceptions = []
for i in range(len(ke1_exceptions)):
    ke1 = np.delete(ke1, (ke1_exceptions[i]), axis=0)
ke10 = ke1.transpose()
def f1(x):
    num = i1
    th_points = np.arctan2(ke10[1][0:num],ke10[0][0:num])
    r_points = np.sqrt(ke10[0][0:num]**2 + ke10[1][0:num]**2)   
    tck = interpolate.splrep(th_points, r_points)
    return interpolate.splev(x, tck)

## Branch 2
ke20 = ke2.transpose()
def f2(x):
    num = i2
    th_points = np.arctan2(ke20[1][0:num],ke20[0][0:num])
    r_points = np.sqrt(ke20[0][0:num]**2 + ke20[1][0:num]**2)   
    tck = interpolate.splrep(th_points, r_points)
    return interpolate.splev(x, tck)

## Branch 3
ke3_exceptions = []
for i in range(len(ke3_exceptions)):
    ke3 = np.delete(ke3, (ke3_exceptions[i]), axis=0)
ke30 = ke3.transpose()
def fm3(x):
    num = j3 #150 #3 - len(ke3_exceptions)
    th_points = np.arctan2(-ke30[1][0:num],ke30[0][0:num]) # minus sign to make th monotonically increasing
    r_points = np.sqrt(ke30[0][0:num]**2 + ke30[1][0:num]**2)
    tck = interpolate.splrep(th_points, r_points)
    return interpolate.splev(x, tck)

## Branch 4
ke40 = ke4.transpose()
def fm4(x):
    num = j4
    th_points = np.arctan2(-ke40[1][0:num],ke40[0][0:num]) # minus sign to make th monotonically increasing
    r_points = np.sqrt(ke40[0][0:num]**2 + ke40[1][0:num]**2)   
    tck = interpolate.splrep(th_points, r_points)
    return interpolate.splev(x, tck)

## Branch 5
ke50 = ke5.transpose()
def f5(x):
    num = len(ke5) -1
    th_points = np.arctan2(ke50[1][0:num],ke50[0][0:num])    #
    r_points = np.sqrt(ke50[0][0:num]**2 + ke50[1][0:num]**2)   
    tck = interpolate.splrep(th_points, r_points)
    return interpolate.splev(x, tck)



#
def  rf(x,y,z,RR0) : return np.sqrt((np.sqrt(x**2+y**2)-RR0)**2 + z**2)

## Cartesian to Polar coords conversion
def cart2pol(xx,yy,zz,RR0):
    rr = np.sqrt((np.sqrt(xx**2+yy**2)-RR0)**2 + zz**2)
    thth = np.arctan2(zz,np.sqrt(xx**2+yy**2)-RR0)
    return rr, thth


## Minima of crossings ~ ~ ~
print('- - - - - - - - - - - - - - - - - - - - - - - - ')
print('Closest points in the branches 1 and 3:')
print('Br1[',i1,'] = ', np.arctan2(ke1[i1][1],ke1[i1][0]), np.sqrt(ke1[i1][0]**2 + ke1[i1][1]**2))
print('Br3[',j3,'] = ', np.arctan2(ke3[j3][1],ke3[j3][0]), np.sqrt(ke3[j3][0]**2 + ke3[j3][1]**2) )
seed1 = np.arctan2(ke3[j3][1],ke3[j3][0])
# print('i1, j3 = ', i1,j3)
# print('dmin(Br1,Br3) = ',  dmin1)




it = 0
th1 = seed1
F1 = f1(th1) - fm3(-th1)
th2 = th1 - 0.01
F2 = f1(th2) - fm3(-th2)
while np.abs(F1) > 1e-4 and it < 5:
    th = th1 - (F1)* (th1-th2)/(F1-F2)
    th2 = th1
    th1 = th
    F2 = F1
    F1 = f1(th1) - fm3(-th1)
    it += 1
# print('th1 = ',th1, ' | f1(th1) = ',f1(th1), 'f3(th1) = ', fm3(-th1))
if np.abs(th1 - seed1) > 0.1:
    th1 = seed1
    
cth = [] # Critical values of theta for the plotting
cth.append(th1) # Crossing of 1st and 3rd branches

print('- - - - - - - - - - - - - - - - - - - - - - - - ')
print('Closest points in the branches 2 and 4:')
print('Br2[',i2,'] = ', np.arctan2(ke2[i2][1],ke2[i2][0]), np.sqrt(ke2[i2][0]**2 + ke2[i2][1]**2) )
print('Br4[',j4,'] = ', np.arctan2(ke4[j4][1],ke4[j4][0]), np.sqrt(ke4[j4][0]**2 + ke4[j4][1]**2) )
seed2 = np.arctan2(ke2[i2][1],ke2[i2][0])
# print('dmin(Br2,Br4) = ',  dmin2)

it = 0
th1 = seed2
# print('F1 = ',F1)
th2 = th1 + 0.01
F2 = f2(th2) - fm4(-th2)
# print('F2 = ',F2)
while np.abs(F1) > 1e-4 and it < 5:
    print('th = ', th)
    th = th1 - (F1)* (th1-th2)/(F1-F2)
    th2 = th1
    th1 = th
    F2 = F1
    F1 = f2(th2) - fm4(-th1)
    it += 1
# print('th2 = ',th1, ' | f2(th2) = ',f2(th1), 'f4(th2) = ', fm4(-th1))
if np.abs(th1 - seed2) > 0.1:
    th1 = seed2

cth.append(th1) # Crossing of 2nd and 4th branches

th3 = np.arctan2(ke5[0][1],ke5[0][0])
cth.append(th3) # 'crossing' of 4th and 5th branches
print('- - - - - - - - - - - - - - - - - - - - - - - - ')


# Grid of initial conditions  > > > > > > > > > > > > > > > > > > > > > > > > 
data= read_data()
R0 = data[7]
opt = int(data[11])
hr = 0.8   # x-axis half-length ('R' or 'y' coordinate)
hr2 = 0.8  # y-axis half-length ('z' coordinate)
z00 = 0.0  # Parameter to for z-axis "base"
region = [ [1, R0-hr, R0 + hr, z00 - hr2, z00 + hr2], #[orb/num,x-.x+,z-,z+]
           [2, R0-hr, R0 + hr, z00      , z00 + hr2], # Upper half poloidal plane
           [1, z00  ,z00 + hr, z00      , z00 + hr2]]

# space sampling #
o_ax = int(data[1])   # Number sample of each axis 
orbits = int(o_ax**2 / region[opt][0]) # Number of grid points (corrected for half plane)

## Grid
y0_list = np.linspace(region[opt][1] , region[opt][2], o_ax)       
z0_list = np.linspace( region[opt][3], region[opt][4], int(o_ax / region[opt][0]))  #
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >

## Testing every point in the grid if it is contained by the branches
ind = []
for i in range(len(y0_list)):
    for j in range(len(z0_list)):
        y0 = y0_list[i] 
        z0 = z0_list[j]
        
        r0 , th0 = cart2pol(0,y0,z0,R0)
        
        if th0 < cth[1]:
            r1 = f1(th0)
            r2 = f2(th0)
            if r0 > r1 and r0 < r2:
                ind.append([i,j,0])
        
        if th0 > cth[1] and th0 < cth[0]:
            r1 = f1(th0)
            r2 = fm4(-th0)
            if r0 > r1 and r0 < r2:
                ind.append([i,j,1])
                
        if th0 > cth[0] and th0 < cth[2]:
            r1 = fm3(-th0)
            r2 = fm4(-th0)
            if r0 > r1 and r0 < r2:
                ind.append([i,j,2])
                
        if th0 > cth[2]:
            r1 = fm3(-th0)
            r2 = f5(th0)
            if r0 > r1 and r0 < r2:
                ind.append([i,j,3])


## Enclosed area computation (from the same regular grid used in CF-data)
# print('Ratio =',len(ind)/(len(y0_list)*len(z0_list)))
# print('Total area = ',hr*hr2*2,'  Total points = ',len(y0_list)*len(z0_list))
print('Enclosed area = ',len(ind)*hr*hr2*2/(len(y0_list)*len(z0_list)))


## FIGURE ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# To use latex fount on figure
plt.rcParams['text.usetex'] = True


## Figure parameters and ploting are selection ~ ~ ~ ~
sz = 6    # markersize
pla = 1   # Plot region selection
af = [[7, 7, -0.85, 0.85, -0.82, 0.82],   # [0] Full poloidal plane
      [10,5.2, -0.82, 0.82, -0.02, 0.82]] # [1] Upper plane

## Figure definition
fig= plt.figure(figsize=(af[pla][0],af[pla][1]))  
ax= fig.gca()


## Enclosed area printing ~ ~ ~
region_color = [(1,1,0,0.2),(0.5,1,0.5,0.2),(0.5,0.5,1,0.2),(1,0.8,0.2,0.2)] # Yellow, Green, Blue, Orange

for k in range(len(ind)):
    y1 = y0_list[ind[k][0]] - R0
    z1 = z0_list[ind[k][1]] 
    ax.plot(y1, z1, 'o', color = region_color[ind[k][2]], markersize= sz)


# Figure - Axes - - - - - - - - - - - - - - - - - - - - - - - 
# ax.set_xlabel('$\~y$', fontsize = 16)                 # Regular matplolib font
# ax.set_ylabel('$\~z$', fontsize = 16 , rotation=0)    # Regular matplolib font
ax.set_xlabel('$\\tilde{y}$',fontsize = 18)             # Latex font
ax.set_ylabel('$\\tilde{z}$',fontsize = 18, rotation=0) # Latex font
ax.set_xlim( af[pla][2] , af[pla][3])  
ax.set_ylim( af[pla][4] , af[pla][5])   


## Branches plotting ~ ~ ~ ~ ~ ~ ~ ~ ~

## Branches ~ orginal data ~ ~ ~
# Br_color = [(1,0.2,0.2),(1,0.4,1),(0.9,0,0.9),(1,0.4,0.2),(0.8,0.2,0.8)] # Red1, Pink1, Pink2, Red2 , Purple1
# ax.plot(ke10[0], ke10[1], 's', markersize= sz , markerfacecolor=(0.3, 0.3, 0.3, 0.2), # 1st Branch 
#              markeredgewidth=2, markeredgecolor= Br_color[0])                          
# ax.plot(ke20[0], ke20[1], 's', markersize= sz , markerfacecolor=(0.3, 0.3, 0.3, 0.2), # 2nd Branch
#              markeredgewidth=2, markeredgecolor= Br_color[1])                        
# ax.plot(ke30[0], ke30[1], 's', markersize= sz , markerfacecolor=(0.3, 0.3, 0.3, 0.2), # 3rd Branch 
#              markeredgewidth=2, markeredgecolor= Br_color[2])                
# ax.plot(ke40[0], ke40[1], 's', markersize= sz , markerfacecolor=(0.3, 0.3, 0.3, 0.2), # 4th Branch
#              markeredgewidth=2, markeredgecolor= Br_color[3])                
# ax.plot(ke50[0], ke50[1], 's', markersize= sz , markerfacecolor=(0.3, 0.3, 0.3, 0.2), # 5th Branch
#              markeredgewidth=2, markeredgecolor= Br_color[4])               

## Branches - splines ~ ~ ~
thar = np.linspace(0,cth[0],100)
ax.plot(f1(thar)*np.cos(thar), f1(thar)*np.sin(thar), '--',color = (0.1,1,0.1), linewidth=3.5)     # 1st Branch

thar2 = np.linspace(0,cth[1],100)
ax.plot(f2(thar2)*np.cos(thar2), f2(thar2)*np.sin(thar2), '--',color = (1,0.1,1), linewidth=3.5)   # 2nd Branch

thar3 = np.linspace(cth[0],np.pi,100)
ax.plot(fm3(-thar3)*np.cos(thar3), fm3(-thar3)*np.sin(thar3), '--',color = (1,0.1,1), linewidth=3.5)   # 3rd Branch

thar4 = np.linspace(cth[1],cth[2],100)
ax.plot(fm4(-thar4)*np.cos(thar4), fm4(-thar4)*np.sin(thar4), '--',color = (0.1,1,0.1), linewidth=3.5) # 4th Branch

thar5 = np.linspace(cth[2],np.pi,100)
ax.plot(f5(thar5)*np.cos(thar5), f5(thar5)*np.sin(thar5), '--',color = (1,0.1,1), linewidth=3.5)       # 5th Branch


## Seed points ~ ~ ~
ax.plot(f1(thar[0])*np.cos(thar[0]), f1(thar[0])*np.sin(thar[0]), 'o',mec = (0.9,0.5,0), mfc = 'c',
        markersize=10,linewidth=1.5)         # 1st Seed point
ax.plot(f2(thar2[0])*np.cos(thar2[0]), f2(thar2[0])*np.sin(thar2[0]), 'o',mec = (0.9,0.5,0), mfc = 'c',
        markersize=10,linewidth=1.5)         # 2nd Seed point
ax.plot(fm3(-thar3[-1])*np.cos(thar3[-1]), fm3(-thar3[-1])*np.sin(thar3[-1]), 'o',mec = (0.9,0.5,0), mfc = 'c',
        markersize=10,linewidth=1.5)         # 3rd Seed point

## Crossing between interpolated branches ~ ~ ~
ax.plot(f1(thar[-1])*np.cos(thar[-1]), f1(thar[-1])*np.sin(thar[-1]), 'o',mec = (0,0.5,0.9), mfc = 'r',
        markersize=10,linewidth=1.5)         # Crossing branch 1 and 3
ax.plot(f2(thar2[-1])*np.cos(thar2[-1]), f2(thar2[-1])*np.sin(thar2[-1]), 'o',mec = (0,0.5,0.9), mfc = 'r',
        markersize=10,linewidth=1.5)         # Crossing branch 2 and 4
ax.plot(f5(thar5[0])*np.cos(thar5[0]), f5(thar5[0])*np.sin(thar5[0]), 'o',mec = (0.9,0.5,0), mfc = 'c',
        markersize=10,linewidth=1.5)         # 4th Seed point
ax.plot(f5(thar5[-1])*np.cos(thar5[-1]), f5(thar5[-1])*np.sin(thar5[-1]), 'o',mec = (0,0.5,0.9), mfc = 'r',
        markersize=10,linewidth=1.5)         # 4Last point of branch 5

## Display figure
plt.show()