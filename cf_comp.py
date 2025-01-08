# Module for vector-tensor computations for "killends_?r_27Nov_gen*.py"
# For the conefield implementation of converse KAM. [27nov23]
#

import numpy as np

from sympy import symbols, sqrt, sin, cos, derive_by_array, Matrix
from sympy import tensorproduct, tensorcontraction
from sympy.utilities.lambdify import lambdify


# Killends criterium # # # #
def cf_computation(r0, th0, ph0,DBf,DBb,e1,m1,n1,e2,m2,n2,R0,w1,w2):
    mp = 0.0
    mm = 0.0
    
    # Read the x(t) 
    r1, th1, ph1 = DBf[0] , DBf[1], DBf[2]  # Forward
    r2, th2, ph2 = DBb[0] , DBb[1], DBb[2]  # Backward
        
    # Read the DB_f(x(t))
    M11, M12, M13 = DBf[3] , DBf[4], DBf[5]
    M21, M22, M23 = DBf[6] , DBf[7], DBf[8]
    M31, M32, M33 = DBf[9] , DBf[10], DBf[11]
    
    # Read the DB_b(x(t))
    N11, N12, N13 = DBb[3] , DBb[4], DBb[5]
    N21, N22, N23 = DBb[6] , DBb[7], DBb[8]
    N31, N32, N33 = DBb[9] , DBb[10], DBb[11]
    
    #Def parameters
    parr = [e1,m1,n1,e2,m2,n2,R0,w1,w2] 

    xi0 = ξ(r0, th0, ph0,R0)     # xi0 
    eta0 = η(r0, th0, ph0,*parr) # eta0
    
    xi1 = ξ(r1, th1, ph1,R0)  #                                 # Forward
    MBf = [[M11, M12, M13],[M21, M22, M23], [M31, M32, M33]]    # Mf
    Mξf = -np.matmul(MBf, xi0)                                  # Mf.xi0  
    sξf  = np.dot(Mξf, np.matmul(β(r1, th1, ph1,*parr), xi1))   # Omega(xi,B, M xi)_f
    Mηf = np.matmul(MBf, eta0)                                  # Mf.etaf
    sηf  = np.dot(Mηf, np.matmul(β(r1, th1, ph1,*parr), xi1))   # Omega(xi,B, M eta)_f
    
    xi2 = ξ(r2, th2, ph2,R0)  #                                     # Backward
    MBb = [[N11, N12, N13],[N21, N22, N23], [N31, N32, N33]]    # Mb
    Mξb = -np.matmul(MBb, xi0)                                  # Mb.xi0   
    sξb  = np.dot(Mξb, np.matmul(β(r2, th2, ph2,*parr), xi2))   # Omega(xi,B, M xi)_b

    Mηb = np.matmul(MBb, eta0)                                  # Mb.eta0
    sηb  = np.dot(Mηb, np.matmul(β(r2, th2, ph2,*parr), xi2))   # Omega(xi,B, M eta)_b
        
    sp = slope(sηf,sξf)
    sm = slope(sηb,sξb)
        
    xiR =  ( xi0[0]*np.cos(th0))/np.sqrt(2*r0) - xi0[1] * np.sin(th0) * np.sqrt(2*r0)
    xiz =  ( xi0[0]*np.sin(th0))/np.sqrt(2*r0) + xi0[1] * np.cos(th0) * np.sqrt(2*r0)
    etaR = (eta0[0]*np.cos(th0))/np.sqrt(2*r0) - eta0[1] *np.sin(th0) * np.sqrt(2*r0)
    etaz = (eta0[0]*np.sin(th0))/np.sqrt(2*r0) + eta0[1] *np.cos(th0) * np.sqrt(2*r0)
         
    mp_0 = mp
    mm_0 = mm
    
    # Cartesian slopes (R,z) plane
    mp = (sp*xiz + etaz ) / (sp*xiR + etaR )  
    mm = (sm*xiz + etaz ) / (sm*xiR + etaR )
    
    return sp,sm, mp, mm, sξf,sξb

# # Slope plot points - (R,z) coords ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# def slope_points_Rz(delta,mm,mp,x1,x2,sg):
#     dx  = delta / np.sqrt((1+mm**2))  # dx**2 + dy**2 = delta**2  #  dy = m dx
#     dx2 = delta / np.sqrt((1+mp**2))
# 
#     p0= [x1,x2]
#     p1= [x1 + dx    , x2 + mm*dx]
#     p2= [x1 - sg*dx2, x2 - sg*mp*dx2]
#     p3= [x1 - dx    , x2 - mm*dx]
#     p4= [x1 + sg*dx2, x2 +sg* mp*dx2]
#     return p1,p2,p3,p4


# Slope computation
def slope(x,y):
    if y != 0 :
        m= x/y
    else:
        m = -1
    return m

# Slope plot points - (R,z) coords ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
def slope_points_Rz(delta,mm,mp,x1,x2,sg):
    dx  = delta / np.sqrt((1+mm**2))  # dx**2 + dy**2 = delta**2  #  dy = m dx
    dx2 = delta / np.sqrt((1+mp**2))

    p0= [x1,x2]
    p1= [x1 + dx    , x2 + mm*dx]
    p2= [x1 - sg*dx2, x2 - sg*mp*dx2]
    p3= [x1 - dx    , x2 - mm*dx]
    p4= [x1 + sg*dx2, x2 +sg* mp*dx2]
    return p1,p2,p3,p4

# Slope plot points - (R,z) coords ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
def slope_points_tr(a,b,mm,mp,th,r,sg):
    mmc = r * (1 + mm * np.tan(th)) / (mm - np.tan(th))
    mpc = r * (1 + mp * np.tan(th)) / (mp - np.tan(th))
    dth = a*b / np.sqrt((b**2 + a**2 * mmc**2))   # BLUE - Upper slope for th^+
    dth2 = a*b / np.sqrt((b**2 + a**2 * mpc**2))  # BLACK - Upper slope for th^+
    
    p0= [th,r]
    p1= [th + dth    , r +     mmc*dth]
    p3= [th - dth    , r -     mmc*dth]
    p2= [th - sg*dth2, r - sg* mpc*dth2]
    p4= [th + sg*dth2, r + sg* mpc*dth2]
    return p1,p2,p3,p4

# Definitions # - - - - - -
π  = np.pi

# Negative identity  - - - 
nI = Matrix([[-1, 0, 0],   
            [0, -1, 0],       
            [0, 0, -1]])



# Cartesian to toroidal coordinates (on a poloidal cross section)
def  rf(x,y,z,RR0) : return np.sqrt((np.sqrt(x**2+y**2)-RR0)**2 + z**2)
def  ψf(x,y,z,RR0) : return ((np.sqrt(x**2+y**2)-RR0)**2 + z**2)/2
# def  ψf2(x,y,z,RR0) : return ((np.sqrt(x**2+y**2))**2 + z**2)/2
def  ψf2(x,y,z,RR0) : return (x**2+y**2+ z**2)/2
def thf(x,y,z,RR0) : return np.arctan2(z,np.sqrt(x**2+y**2)-RR0)
def thf2(x,y,z,RR0) : return np.arctan2(z,y)

# ## Tensor algebra # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# operations
def dotproduct(A,B) : return tensorcontraction(tensorproduct(A,B),(0,1)) # dot vectors A, B
def multiply(A,B)   : return tensorcontraction(tensorproduct(A,B),(1,2)) # multiply matrix A with vector B
def contract(A,B)   : return tensorcontraction(tensorproduct(A,B),(2,3)) # contract 3-form A with vector B
# 3D Levi-Civita symbol (for volume form)
def LC(i,j,k) : return(j-i)*(k-i)*(k-j)/2







## SYSTEM ##

# Symbolic variables for algebraic computations - - - - - - - - - - 
r, th, ph = symbols('r, th, ph')
Vr, Vth, Vph = symbols('Vr, Vth, Vph')

# Derivative Matrix variables
Brr,Brt,Brp = symbols('Brr,Brt,Brp')
Btr,Btt,Btp = symbols('Btr,Btt,Btp')
Bpr,Bpt,Bpp = symbols('Bpr,Bpt,Bpp')

# Symbolic paramters
E1,M1,N1,E2,M2,N2,RR0,W1,W2 = symbols('E1,M1,N1,E2,M2,N2,RR0,W1,W2 ')
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# Magnetic field # = = = = = = = = = = = = = = = = = = = = = = = = = = 
def B(ψ,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2) :
    Bψ  = (M1*E1*(ψ**(M1/2))*(ψ - RR0**2)*sin(M1*th-N1*ph) + M2*E2*(ψ**(M2/2))*(ψ - RR0**2)*sin(M2*th-N2*ph))  #*R0/R**2 #/ (r*R) | *R0/R**2
    Bth = (W1 + 2*W2*ψ + E1*(ψ**(M1/2))*(((ψ - RR0**2)*M1/(2*ψ)) + 1)*cos(M1*th-N1*ph)
           + E2*(ψ**(M2/2))*(((ψ - RR0**2)*M2/(2*ψ)) + 1)*cos(M2*th-N2*ph)) #* R0/R**2 # / (r*R) | *R0/R**2
    Bph = 1    # R0/R**2 # 1 /(R*r) | *R0/R**2
    return [Bψ, Bth, Bph]

# flow #
def f(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2) : return B(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2)

# Jacobian #
def Df(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2) :
    k = len(f(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2))
    return [derive_by_array(f(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2)[i],[r,th,ph]) for i in range(0,k)]


# Derivative Matrix #
def M(r,th,ph,Brr,Brt,Brp,Btr,Btt,Btp,Bpr,Bpt,Bpp,E1,M1,N1,E2,M2,N2,RR0,W1,W2):
    M0 = [[Brr,Brt,Brp],[Btr,Btt,Btp],[Bpr,Bpt,Bpp]]
    M1 = multiply(Df(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2), M0)
    return [M1[0][0],M1[0][1],M1[0][2],M1[1][0],M1[1][1], M1[1][2],M1[2][0],M1[2][1],M1[2][2],0,0,0,0,0,0,0,0,0 ]


# whole system - Killends (27Nov - Derivative Matrix integration) # # # # # # # # # # #
def H(r,th,ph,Brr,Brt,Brp,Btr,Btt,Btp,Bpr,Bpt,Bpp,E1,M1,N1,E2,M2,N2,RR0,W1,W2) :
    return [*f(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2),
            *M(r,th,ph,Brr,Brt,Brp,Btr,Btt,Btp,Bpr,Bpt,Bpp,E1,M1,N1,E2,M2,N2,RR0,W1,W2)]

Hnum = lambdify((r,th,ph,Brr,Brt,Brp,Btr,Btt,Btp,Bpr,Bpt,Bpp,E1,M1,N1,E2,M2,N2,RR0,W1,W2),
                H(r,th,ph,Brr,Brt,Brp,Btr,Btt,Btp,Bpr,Bpt,Bpp,E1,M1,N1,E2,M2,N2,RR0,W1,W2), 'numpy')

def system3(t, X):
    x1,x2,x3, b11,b12,b13, b21,b22,b23,b31,b32,b33, ee1,mm1,nn1,ee2,mm2,nn2,rr0,ww1,ww2 = X
    dXdt = Hnum(x1,x2,x3, b11,b12,b13, b21,b22,b23,b31,b32,b33, ee1,mm1,nn1,ee2,mm2,nn2,rr0,ww1,ww2)
    return dXdt
## = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 


## DIRECTION FIELD ξ ##

# Adapted cartesian coordinates' metric
def metric(r,RR0):
    return Matrix([[1/(2*r), 0, 0],  [0, 2*r, 0],   [0, 0, RR0**2]])


def  J(r,th,ph) : return r
def dJ(r,th,ph) : return derive_by_array(J(r,th,ph),[r,th,ph])

def xi(r,th,ph,RR0) :
    g = metric(r,RR0)
    gradJ = multiply(g.inv(), dJ(r,th,ph))
    #gradHH = multiply(metric.inv(), dHH(x,y,px,py))
    #dH2  = dotproduct(gradH, dHH(x,y,px,py))
    #dHdJ = dotproduct(gradJ, dHH(x,y,px,py))
    #a = dHdJ/dH2
    return gradJ #- a*gradH

ξ = lambdify((r,th,ph,RR0), xi(r,th,ph,RR0), 'numpy')

def mod2ξ(r,th,ph,RR0) : return np.dot(ξ(r,th,ph,RR0), ξ(r,th,ph,RR0))




## Horizontal auxiliary field (Ortogonal to B and xi, and then projected in the poloidal plane) ##
def eta(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2) :                 
    g = metric(r,RR0)
    f0 = f(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2)
    f1  = multiply(g, f(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2)) # f1 = B^\flat
    f2 = [0, -f1[2]/(2*r*RR0), f1[1]/(2*r*RR0)] # f2 = -B x xi = xi x B
    f3 = dotproduct(f2, multiply(g, f2)) # |f2|^2
    eta0 =          - (f2[2]/f3) * f0[0]  # Using B to remove eta^phi component  
    eta1 = f2[1]/f3 - (f2[2]/f3) * f0[1]  # (f2 - c B)^j   s.t.  (f2 -c B)^phi = 0  
    return [eta0, eta1, 0]
#     return f2

η = lambdify((r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2), eta(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2), 'numpy') # B . xi x eta > 0



## CONVERSE KAM CONDITION ##

# volume form #
def vol(r,th,ph,RR0) :
    ρ = RR0**2  # 
    v = ρ*np.fromfunction(LC, (3,3,3))
    return v.tolist()

# magnetic flux form #
def beta(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2) : return contract(vol(r,th,ph,RR0), B(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2))
β = lambdify((r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2), beta(r,th,ph,E1,M1,N1,E2,M2,N2,RR0,W1,W2), 'numpy')


