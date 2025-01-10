"""
 Script to plot files "cf_[ini_points]_[tf]_[e1]_[m1]_[n1]_[e2]_[m2]_[n2].txt"
 from  'cKAM_CF_2r.py'
 
 [Implementation of converse KAM result (MacKay'18) to an integrable
 magnetic field example: axisymmetrical magnetic field perturbed with
 2 helical modes (Kallinikos'14).]   ~~ Conefield approach (Martinez '25) ~~

 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 Inputs:  'ini.txt' - Parameter file
            'ke_{orbits}_{tf}_{ep1}_{m1}_{n1}_{ep2}_{m2}_{n2}.txt'
          from 'conefield_2r_gen.py'

 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 Author: David Martinez del Rio (U.Warwick)

 Based on the orignal codes by:
          Nikos Kallinikos (U.Warwick, U. W.Macedonia, Int Hellenic U.)
          David Martinez del Rio (U.Warwick)
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import simplejson
#import os
from matplotlib.colors import LinearSegmentedColormap

from cf_tools import *  # Data format manipulation, and graphic - subroutines/definitions

# To use \tex on axis fonts
plt.rcParams['text.usetex'] = True

## SET RUN PARAMETERS ##
data = read_data()

tf = int(data[0])
Pts= int(data[1])


## IMPORTS DATA ##
#ep=data[2]
m1 = data[3]
n1 = data[4]
w1 = data[5]
w2 = data[6]
R0 = data[7]
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

## Area studied by 'cKAM_CF_2r.py'  
hr = 0.8  # Half of R-axis
hr2 = 0.8 # z-axis

e1=data[2]
ep1 = ep2str(e1) # Converted to string for file manipulation 
e2=data[8]
ep2 = ep2str(e2) # Converted to string

## OPENS FILE ##
file2 = open('cf_%s_%s_%s_%s_%s_%s_%s_%s.txt' % (points, tf,ep1, int(m1),int(n1),ep2, int(m2),int(n2)), 'r')
result = simplejson.load(file2)
file2.close()



## Data read / variable allocation - - - - - - - - - - - - - - - - - - - - -  
x1p = [] # Red point
x2p = [] # ~ eliminated points

x1u = [] # Blue points
x2u = [] # ~ not eliminated points 

x1 = []  # y-R0 value in (y,z)
x2 = []  # z value in (y,z)
q  = []  # te/tf : relative time of detection
mp = []  # upper slope value
mm = []  # Lower slope bound
thc = [] # Angle value of the (x10,y10) cartesian point
rc = []  # Radial value of the (x10,y10) cartesian point
nf = []  # Boundary points index
btr=0    # Blue-tp-Red boundary identifier variable
tf2 = int(tf/2) # Half of integration time
for i in range(len(result)) :
    x10 = result[i][0] - R0 
    x20 = result[i][1]
    ie  = result[i][2]
    if abs(x10) + abs(x20)> 0.05:
        te  = result[i][3]
    else:
        te = tf2
    mm0  = result[i][5]
    mp0  = result[i][6]
    if ie == 1 :
    #if ie == 1 and abs(x10) + abs(x20)> 0.1:
        x1p.append(x10)   # Red points
        x2p.append(x20)   #
        btr +=1
        if btr ==1 and x20 >0 and i >0:
            #nf.append(i) # Inner (island) point
            nf.append(i-1) # Outer (island) point
    else :
        x1u.append(x10)   # Blue points
        x2u.append(x20)   #
        if btr>0:
            #nf.append(i-1) # Inner (island) point
            nf.append(i)   # Outer (island) point
            btr = 0
    x1.append(x10)
    x2.append(x20)
    q.append(te/tf2)
    r0 = np.sqrt(x10**2 + x20**2)
    rc.append(r0)
    thc.append(np.arctan2(x20,x10))
    mm.append(mm0)   # Last slopes computed
    mp.append(mp0)   #


# Island - BNDRY
# Simple algorithm to find the boundary | Boundary: Blue points near a Red point = = =
bnd = []
compl = []
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
    if q[i]== 1:
        compl.append(i)


## Area   ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ 
# print('Total points = ', points)           #  
# print('Ratio = ', len(x1p)/points)         #  
print('Area = ', len(x1p)*2*hr*hr2/(points)) # 


## FIGURES ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

## FIGURES parameters ~ ~ ~ ~ 
sz = 5 # Marker size in all plots
# pla = 1
pla2 = 0
# Plotting area selection ~ ~ ~ ~ 
#      factor  xl1    xl2    yl1     yl2      [pla]
af = [[ 7, 7, -0.85,  0.85, -0.82,  0.82],  # [0] Full poloidal plane
      [10,5.2, -0.82, 0.82, -0.02,  0.82],  # [1] Upper plane 
      [8, 8,  0.1,  0.19,  0.43,  0.52],    # [2] Upper plane ~ zoom 1
      [8, 8,  0.312,  0.402,  -0.002,  0.088], # [3] Upper plane ~ zoom 2
      [8, 8,  0.472,  0.562,   0.23,  0.32], # [4] Upper plane ~ zoom 3
      [ 7, 7, -0.02,  0.6,  -0.02,  0.52]]   # [5]Positive upper quadrant

#      factor     xl1    xl2    yl1   yl2      [pla2]
af2 = [[10, 5.2, -0.05 , np.pi, 0,    0.85], # [0] "Upper half" plane
       [10, 8,    1.25 , 1.82,  0.41, 0.55], # [1] Zoom 3
       [10, 8,    0.8 ,  1.4,   0.58, 0.70], # [2] Zoom 2
       [10, 8,    0.2 ,  1.2,   0.3,  0.82]] # [3] Zoom 1


# FIGURES definition ~ ~ ~ ~ 
fig= plt.figure(figsize=(af[pla][0],af[pla][1]))   # Red-Blue
ax= fig.gca()

fig2= plt.figure(figsize=(af[pla][0]+2,af[pla][1])) # Hue-figure
ax2= fig2.gca()

fig3= plt.figure(figsize=(af2[pla2][0],af2[pla2][1])) # Hue-figure
ax3= fig3.gca()




## PLOTTING ##

# standard Converse KAM plot - (RED-BLUE) - - - - FIGURE 1 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

ax.plot(x1p, x2p, 'ro', markersize= sz)
ax.plot(x1u, x2u, 'bo', markersize= sz)

# Figure - Axes ~ ~ ~ ~ 
# ax.set_xlabel('$\~y$', fontsize = 16)                 # Regular matplolib font
# ax.set_ylabel('$\~z$', fontsize = 16 , rotation=0)    # Regular matplolib font
ax.set_xlabel('$\\tilde{y}$',fontsize = 18)             # Latex font
ax.set_ylabel('$\\tilde{z}$',fontsize = 18, rotation=0) # Latex font

ax.set_xlim( af[pla][2] , af[pla][3])  
ax.set_ylim( af[pla][4] , af[pla][5])   

## OPTIONAL: Boundary plot on Figure 1 ~ ~ ~ ~ 
for j in range(0,len(bnd),1):
    nn = bnd[j]
    ax.plot(x1[nn], x2[nn], 'go', markersize= sz +2)
    
    
# Converse-KAM-speed plot - (HUES) ~ ~ ~ ~ ~ ~ ~ ~  FIGURE 2  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
sc = ax2.scatter(x1, x2, c=q, vmin=0, vmax=1, s=10*int(sz/3), cmap = cm1) #
fig2.cbar = fig2.colorbar(sc)
fig2.cbar.set_label('$q$', rotation=0, fontsize = 16, labelpad=10, y=0.5) # 

# FIGURE 2 : Slopes plots ~ ~ ~ ~ ~ ~ ~ ~ 
nnskip = 1 # Number of points in the boundary to have their slopes plotted 1 every [nnskip]
delta = 0.004  # Slopes size - Fig2

sec = 0 # Cone plotting region: 0 Boundary "∂S" | 1: Complement "S^c"
if sec == 0:
    region = bnd
else:
    region = compl

for j in range(0,len(region),nnskip):  
    nn = region[j]                     

    # BOUNDARY POINTS ~ ~ ~ ~ ~ ~ ~ ~ 
#     ax2.plot(x1[nn], x2[nn], 'rp', markersize= sz +2)

    # SLOPES ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    dx = delta / np.sqrt((1+mm[nn]**2))  # dx**2 + dy**2 = delta**2  #  dy = m dx
    ax2.plot([x1[nn] - dx,x1[nn] + dx],[x2[nn] - mm[nn]*dx,x2[nn] + mm[nn]*dx],'gs-', linewidth=2, markersize=2)

    dx2 = delta / np.sqrt((1+mp[nn]**2))
    ax2.plot([x1[nn] - dx2,x1[nn] + dx2],[x2[nn] - mp[nn]*dx2,x2[nn] + mp[nn]*dx2],'ms-', linewidth=2, markersize=2)

    # Shaded area between SLOPES ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    sg=1
    if mm[nn]< 0 and x1[nn] > 0:
        sg = -1
    if mp[nn]> 0 and x1[nn] < 0:
        sg = -1
    p0= [x1[nn],x2[nn]]
    p1= [x1[nn] + dx , x2[nn] + mm[nn]*dx]
    p2= [x1[nn] - sg*dx2, x2[nn] - sg*mp[nn]*dx2]
    p3= [x1[nn] - dx , x2[nn] - mm[nn]*dx]
    p4= [x1[nn] + sg*dx2, x2[nn] +sg* mp[nn]*dx2]

    # Green shaded region in the acceptable cone region
    tp = np.array([p0,p1,p2,p0,p3,p4])

    if q[nn] == 1:
        tp = np.array([p0,p1,p2,p0,p3,p4])
        farbe = (0.3, 0.8, 0.3, 0.3)
    else:
        tp = np.array([p0,p2,p1,p0,p4,p3])
        farbe = (0.8, 0.3, 0.3, 0.3)
    t1 = plt.Polygon(tp, color=farbe)
    ax2.add_patch(t1)



# FIGURE 2 - Axes ~ ~ ~ ~ ~ ~ ~ ~ 
# ax2.set_xlabel('$\~y$', fontsize = 16)                 # Regular matplolib font
# ax2.set_ylabel('$\~z$', fontsize = 16 , rotation=0)    # Regular matplolib font
ax2.set_xlabel('$\\tilde{y}$',fontsize = 18)             # Latex font
ax2.set_ylabel('$\\tilde{z}$',fontsize = 18, rotation=0) # Latex font

ax2.set_xlim( af[pla][2] , af[pla][3])  
ax2.set_ylim( af[pla][4] , af[pla][5])  



# - - FIGURE 3 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# Converse-KAM-speed plot - (HUES - th,r) ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
delta=0.05
ax3.set_xlim( af2[pla2][2] , af2[pla2][3])  
ax3.set_ylim( af2[pla2][4] , af2[pla2][5]) 
#          #
ax3.set_xlabel('$\\vartheta$', fontsize = 16)                # Axes names @Fig3 #
ax3.set_ylabel('$\sqrt{2\psi}$', fontsize = 16 , rotation=0) #                  #

sc3 = ax3.scatter(thc, rc, c=q, vmin=0, vmax=1, s=10*int(sz/3), cmap = cm1) #
fig3.cbar = fig3.colorbar(sc3)
fig3.cbar.set_label('$q$', rotation=0, fontsize = 16, labelpad=10, y=0.5) # 


# FIGURE 3 : Slopes plots ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
aa = 0.01   # Fig3 - Slope sizes - small
bb = 0.005  #

for j in range(0,len(bnd),nnskip):  # Plot boundary points and slopes - - - - - 
    nn = bnd[j]
    # BOUNDARY POINTS ~ ~ ~ ~ ~ ~ ~ ~ 
    ax3.plot(thc[nn], rc[nn], 'rp', markersize = sz)
    
    # SLOPES ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    mmc = rc[nn] * (1 + mm[nn]*np.tan(thc[nn])) / (mm[nn] - np.tan(thc[nn]))
    mpc = rc[nn] * (1 + mp[nn]*np.tan(thc[nn])) / (mp[nn] - np.tan(thc[nn]))
    
    dth = aa*bb / np.sqrt((bb**2 + aa**2 * mmc**2))   # Green - Upper slope for th^+
    ax3.plot([thc[nn] - dth, thc[nn] + dth],[rc[nn] - mmc*dth, rc[nn] + mmc*dth],'gs-', linewidth=2, markersize=2)
 
    dth2 = aa*bb / np.sqrt((bb**2 + aa**2 * mpc**2))  # Magenta - Upper slope for th^+
    ax3.plot([thc[nn] - dth2,thc[nn] + dth2],[rc[nn] - mpc*dth2,rc[nn] + mpc*dth2],'ms-', linewidth=2, markersize=2)

    # Shaded area between SLOPES ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    sg=-1
    if mmc > 0:
        sg = -1
    p0= [thc[nn]       ,    rc[nn]]
    p1= [thc[nn] + dth ,    rc[nn] +      mmc* dth]
    p2= [thc[nn] - sg*dth2, rc[nn] - sg * mpc*dth2]
    p3= [thc[nn] - dth ,    rc[nn] -      mmc* dth]
    p4= [thc[nn] + sg*dth2, rc[nn] + sg * mpc*dth2]

    tp = np.array([p0,p1,p2,p0,p3,p4])

    if q[nn] == 1:
        tp = np.array([p0,p1,p2,p0,p3,p4])
        farbe = (0.3, 0.8, 0.3, 0.3)  # GREEN shade
    else:
        farbe = (0.8, 0.3, 0.3, 0.3)  # RED shade
    t2 = plt.Polygon(tp, color=farbe)
    ax3.add_patch(t2)


## Save figures to file - - - - - - -
# fig.savefig('2r_RB_%s_%s_%s_%s_%s_%s_Rz_C_%s.png' % (int(m1),int(n1),ep1, int(m2),int(n2),ep2,tf), dpi=300)
# fig2.savefig('2r_hue_%s_%s_%s_%s_%s_%s_Rz_C_%s.png' % (int(m1),int(n1),ep1, int(m2),int(n2),ep2,tf), dpi=300)
# fig3.savefig('2r_hue_%s_%s_%s_%s_%s_%s_thr_C_%s.png'% (int(m1),int(n1),ep1, int(m2),int(n2),ep2,tf), dpi=300)
# fig2.savefig('C1_%s_Rz_%s_zoom%s.png' % (ep2,tf,pla-1), dpi=300)
# fig2.savefig('C1_%s_Rz_%s_zoom%sA.png' % (ep2,tf,pla-1), dpi=300)
# fig3.savefig('zoom_%s_thr_%s.png'% (ep2,tf), dpi=300)


# PLOT THE FIGURES - - - - - - - - -
plt.show()