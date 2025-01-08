"""
 Conefiled implementation of converse KAM result (MacKay'18) to an integrable
 magnetic field example: axisymmetrical magnetic field perturbed with
 2 helical modes (Kallinikos'14).
 
 ~ ~ Conefield implementation (Martinez'25) ~ ~
 The script computes the confield for a regular grid of initial conditions taken
 over the poloidal plane phi =0 in symplectric coordinates, for toroidal magnetic
 field transverse to the section. 

 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 Input:  'ini.txt' - Parameter file
 Output: "cf_{orbits}_{tf}_{ep1}_{m1}_{n1}_{ep2}_{m2}_{n2}.txt"
 
 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 Author: David Martinez del Rio (U.Warwick)
         based in the previous Converse KAM codes created by N Kallinikos and
         D Martinez-del-Rio - PPCF(2023).
"""
import matplotlib.pyplot as plt
import numpy as np

from sympy import symbols, sqrt, sin, cos, derive_by_array, Matrix
from sympy import tensorproduct, tensorcontraction
from sympy.utilities.lambdify import lambdify
from scipy.integrate import solve_ivp

from joblib import Parallel, delayed

import time
import datetime
import simplejson

from cf_tools import *  # Data format manipulation, and graphic - subroutines/definitions
from cf_comp import *   # Conefield computation and tensor (symbolic & numerical) subroutines/funtions


start = time.time()




## PRELIMINARIES ##
# Parameter values capture - - - - -
data= read_data()
e1 = data[2]
m1 = data[3]
n1 = data[4]
w1 = data[5]
w2 = data[6]
R0 = data[7]

e2 = data[8]
m2 = data[9]
n2 = data[10]

#String formating for labels - 
ep1 = ep2str(e1)
ep2 = ep2str(e2)
# - - - - - - - - - - - - - - - - - -

# Definitions # = = = = = = = = = = 
π  = np.pi

# Negative identity  - - - 
nI = Matrix([[-1, 0, 0],   
            [0, -1, 0],       
            [0, 0, -1]])

# Adapted coordinates - - - -- 
metric = Matrix([[1/(2*r), 0, 0],   # Adapted cartesian 
                 [0, 2*r, 0],       # coordinates
                 [0, 0, R0**2]])
# = = = = = = = = = = = = = = = = =

## SYSTEM ##

# Symbolic variables for algebraic computations
r, th, ph = symbols('r, th, ph')
Vr, Vth, Vph = symbols('Vr, Vth, Vph')

# Derivative Matrix variables
Brr,Brt,Brp = symbols('Brr,Brt,Brp')
Btr,Btt,Btp = symbols('Btr,Btt,Btp')
Bpr,Bpt,Bpp = symbols('Bpr,Bpt,Bpp')

# Symbolic paramters
E1,M1,N1,E2,M2,N2,RR0,W1,W2 = symbols('E1,M1,N1,E2,M2,N2,RR0,W1,W2 ')




            

## INTEGRATION & RESULTS ##

# time parameters #
tf = int(data[0]/2)  # TIME OUT ##
t_int_b = [0, -tf]
t_int_f = [0, tf]

opt = int(data[11]) # REGION SELECTION ##
# [0] - whole plane 
# [1] - upper plane
# [2] - positive quadrant
hr = 0.8   # x-axis half-length ('R' or 'y' coordinate)
hr2 = 0.8  # y-axis half-length ('z' coordinate)
z00 = 0.0  # Parameter to for z-axis "base"
region = [ [1, R0-hr, R0 + hr, z00 - hr2, z00 + hr2], #[orb/num,x-.x+,z-,z+]
           [2, R0-hr, R0 + hr, z00      , z00 + hr2],
           [1, z00  ,z00 + hr, z00      , z00 + hr2],
           [1, R0 + 0.3  ,R0 + 0.6, 0.2      , 0.5]]

# space sampling #
o_ax = int(data[1])   # sample of each axis (unless opt = 2, Upper plane)
#o_ax = 2   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
orbits = int(o_ax**2 / region[opt][0]) # Number of grid points

y0_list = np.linspace(region[opt][1] , region[opt][2], o_ax)       
z0_list = np.linspace( region[opt][3], region[opt][4], int(o_ax / region[opt][0]))  #

ph0 = 0


def loop(i,j) :
    
    s = time.time()
    
    #y0 = y00
    y0 = y0_list[i] # No longer physical coordinates but
    z0 = z0_list[j] # x = sqrt(2ψ) cos(vth), y = sqrt(2ψ) sin(vth)


    ψ0 = ψf(0,y0,z0,R0)
    vth0 = thf(0,y0,z0,R0)  
    
    
    # Killends - 27Nov - Derivative Matrix - - DB - - DB - - DB - - DB - - DB - - DB - - DB - - 
    B0 = [1,0,0,0,1,0,0,0,1]  # v0 = ξ = (r0,0,0)
    par = [e1,m1,n1,e2,m2,n2,R0,w1,w2] 
    Z0 =  [ψ0, vth0, ph0,*B0,*par]
    mp = 0.0
    mm = 0.0
    
    
    SOLDBf = solve_ivp(system3, t_int_f, Z0, method='RK45', rtol=1e-9, atol=1e-11)
    SOLDBb = solve_ivp(system3, t_int_b, Z0, method='RK45', rtol=1e-9, atol=1e-11)
    
    
    #print('len(SOLDBf.t), len(SOLDBb.t) = ',len(SOLDBf.t),'  ',len(SOLDBb.t))
    t_test = len(SOLDBf.t)
    if len(SOLDBf.t) > len(SOLDBb.t):
        t_test = len(SOLDBb.t)

    DBf = np.transpose(SOLDBf.y)
    DBb = np.transpose(SOLDBb.y)
    
    for k in range(0,t_test):
        
        sp1,sm1,mp, mm, sξf,sξb = cf_computation(ψ0, vth0, ph0,DBf[k],DBb[k],*par)
#         sp = sp1
#         sm = sm1
        if k <= 5:
            sp = sp1
            sm = sm1
        if k>5 and sp1 < sp:
            sp = sp1
        if k>5 and sm  < sm1:
            sm = sm1
            
        if sp < sm or sξf < 0 or sξb > 0:
            te = SOLDBf.t[k]
            ie = 1.0
            #print('ie , te =',ie,' , ',te)
            break
        else :
            te = tf
            ie = 0.0
        
        
    # - - DB - - DB - - DB - - DB - - DB - - DB - - DB - - DB - - DB - - DB - - DB - - DB - - 
    
    f = time.time()
    tr = f-s
    

    return y0, z0, ie, te, tr, mm, mp #, mm_0, mp_0


## PARALELIZATION ON CPU CORES ##
if __name__ == "__main__":
    result = Parallel(n_jobs=-1)(delayed(loop)(i,j) for i in range(o_ax) for j in range(int(o_ax / region[opt][0])))
                           # -1=all cores, 1=serial


## RUNNING TIME ##
finish = time.time()
hours, rem = divmod(finish-start, 3600)
minutes, seconds = divmod(rem, 60)
print('time = {:0>2}:{:0>2}:{:02.0f}'.format(int(hours),int(minutes),seconds))


## SAVING DATA ##
file = open('cf_%s_%s_%s_%s_%s_%s_%s_%s.txt' % (orbits, 2*tf, ep1,int(m1),int(n1), ep2,int(m2),int(n2)),'w')
simplejson.dump(result, file)
file.close()



