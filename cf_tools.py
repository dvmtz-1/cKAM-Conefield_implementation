# Module for edition subroutines for "killends_?r_27Nov_gen*.py"
# For the conefield implementation of converse KAM. [27nov23]
#

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

## Data manipulation ##

# Read parameter file - - - - - - 
def read_data():
    file1 = open("ini.txt")
    data = file1.readlines()
    r = []
    for line in data:
        r.append(float(line))
    file1.close()
    return r


# Read Orbit data from "cKAM_CF_2r.py"
def read_data4(file):
    r = np.loadtxt(file+".txt", unpack=False)
    return r

# String edition for files - - - 
def ep2str(ep):
    if ep < 0.1 and ep >= 0.01:
        eps = '0'+str(int(ep*1000))
    elif ep >= 0.001:
        eps = '00'+str(int(ep*10000))
    else:
        eps = '000'+str(int(ep*100000))
    return eps


## Printing-to-screen subroutines ##

# Screen printing spacing
def space(n):
    esp1 = ' '
    esp = ''
    m = 0
    if abs(n)>=100:
        m += 0
    elif abs(n) >= 10:
        m += 1
    if n >= 0 :
        m += 1
    if abs(n) < 10:
        m += 2
    for i in range(m):
        esp = esp + esp1
    return esp



# Graphic definitions ~ ~ ~

## Adapted color-map  - - - - - - - ##
col_list0 = [(0, 0, 1),
   ( 0.2752 ,   0.5037 ,   0.9677),
(    0.2287  ,  0.6360  ,  0.9896),
 (   0.1382  ,  0.7637   , 0.8945),
  (  0.0929 ,   0.8690    ,0.7571),
   ( 0.1732 ,   0.9403,    0.6200),
    (0.3561 ,   0.9856 ,   0.4452),
(    0.5574  ,  0.9988  ,  0.2882),
 (   0.7077,    0.9718   , 0.2099),
  (  0.8380 ,   0.9022    ,0.2088),
   ( 0.9394  ,  0.8040,    0.2275),
(    0.9907   , 0.6892 ,   0.2088),
 (   0.9917,    0.5416  ,  0.1496),
  (  0.9527 ,   0.3849   , 0.0822),
    (1,0,0)]
myorder = [14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
col_list = [col_list0[i] for i in myorder]
cmap_name = 'my_list'
cm1 = LinearSegmentedColormap.from_list(cmap_name, col_list, N=90)