# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:49:24 2019

@author: dimo1
"""

import SAStool
import pdf2iq

import numpy as np
import pickle

from os import makedirs
from scipy import stats
from scipy.optimize import minimize

from matplotlib import pyplot as plt
from datetime import datetime

plt.rcParams['font.size']=12

#%%
#default configs
method='BFGS' #see scipy.optimize.minimize for more options
mode='fast' #or 'fast' 'medium' 'slow'
fend='out' #or 'pickle' 'out'

#input_fname='sphere.pickle'
input_fname="35AspherePD025.out"
s=0.25 #standard deviation of particle size distribution (P/D ratio)
m=50 #mean radius of particle size distribution (anstrom)
dist='lognorm' # or 'normal'

R_size=51 #number of datapoints for single-particle PDDF

save=True #or False
output_fname=None

#%% 
#functions

def synthetic(pddf_norm,distf,Rrange):
    pddf_syn=np.zeros(Rrange.size)
    for R in Rrange:
        pddf_y=pddf_norm*(R**6);
        pddf_xy=np.interp(Rrange,np.linspace(0,2*R,pddf_y.size),pddf_y)
        pddf_syn=pddf_syn+pddf_xy*distf.pdf(R)
    return pddf_syn/pddf_syn.max()

def rmse(y1,y2):
  
    y1=y1+y1.mean()
    y2=y2+y2.mean()
    y2=y2*y1.sum()/y2.sum()

    y2=np.interp(np.linspace(0,1,y1.size),np.linspace(0,1,y2.size),y2)
    a=(y2 - y1)**2 / y1

    return a.sum()/y1.size

#define the goal function for minimization
def goal(x0,pdf_sync,distf,R_trim):
    x0[0]=0
    x0[-1]=0
    global S1
    CHI=rmse(pdf_sync,synthetic(x0,distf,R_trim))
    POS=POSITV(x0)
    SMO=SMOOTH(x0)
    GOAL=1*CHI+1*( (1-POS)/0.001 ) + 1*0.001*((SMO-1.1))
#The coefficients of the goal function is tunable
    return GOAL

def POSITV(x0):
    return np.linalg.norm(x0[x0>0])/np.linalg.norm(x0)

def SMOOTH(x0):
    return np.linalg.norm(np.diff(x0)*x0.size) / (np.linalg.norm(x0)*np.pi)
  
def printnow(xk):
    global N_iter
    global S1
    global temp1
    global temp2
    plt.clf() 
    GOAL=goal(xk,pdf_sync,distf,R_trim)
    CHISQR=rmse(pdf_sync,synthetic(xk,distf,R_trim))
    POS=POSITV(xk)
    print ('\t%i\t%.4E\t%.6f\t%.6f\t%.6E\t%.1f%%' % (N_iter, GOAL, POS, SMOOTH(xk), CHISQR, (1-CHISQR/GOAL)*100) )
    plt.subplot(121)
    plt.plot(R_space,xk/xk.max(),'o')
    temp1=np.concatenate((temp1,xk/xk.max()),axis=0)
    # plt.xlabel(r'$r/D_{max}$')
    # plt.ylabel(r'$PDDF(r)$')
    plt.subplot(122)
    plt.plot(R_trim, synthetic(xk,distf,R_trim),'o')
    temp2=np.concatenate((temp2,synthetic(xk,distf,R_trim)),axis=0)
    plt.plot(R_trim, pdf_sync,':')
    # plt.xlabel(r'$r$/$\AA$')
    # plt.ylabel(r'$PDDF(r)$')
    plt.show()
    plt.pause(0.05)
    N_iter=N_iter+1

def procnow():
    ax=plt.gca()
    fig = plt.gcf()
    plt.tick_params(labelsize=18)
    #plt.xlabel(fontsize=18, fontweight='bold')
    #plt.ylabel(fontsize=18, fontweight='bold')
    plt.yticks(fontsize='16')
    plt.xticks(fontsize='16')
    plt.tick_params(direction='in',length=6,width=2)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    fig.set_size_inches(4,3)
    fig.set_dpi(150)
    plt.tight_layout()
    plt.show()
    #plt.savefig('figure1.png',dpi=300)

#%%
#data input
    
#read GNOM output
if fend=='out':
    data=SAStool.readpddf(input_fname)
    R=data[:,0]
    pdf_sync=data[:,1]

#read SAXScraft binary pickle outputs
if fend=='pickle':
    with open(input_fname, 'rb') as f:
        pdf=pickle.load(f)
    pdf=pdf/max(pdf)
    stddev = s
    mean = m
    R=np.linspace(0,3*mean+0,int((3*mean)/3+1))
    distf=stats.lognorm(stddev,scale=mean)
    pdf_sync=synthetic(pdf,distf,R)
    pdf_sync=pdf_sync/pdf_sync.max()
    
    q,iq0=pdf2iq.pdf2iq(np.linspace(0,1,pdf.size),pdf,mean=mean,q_min=0.001,q_max=1,r_size=2000)
    
#reduce the data points in the real space
R_trim=np.linspace(0,R.max()+0,int((R.max())/3+1))
pdf_sync=np.interp(R_trim,R,pdf_sync)

#normalize the real space data and (optional) add random pertubations
pdf_sync=pdf_sync/pdf_sync.max()
pdf_sync=pdf_sync*(1+0.00*np.random.normal(loc=0.0,scale=1.0,size=pdf_sync.size))

#the level of pertubations are tunable

#%%

#initial guess of the single particle PDDF
R_space=np.linspace(0,1,R_size)
x=stats.norm.pdf(R_space,0.5,0.2)
x=x/max(x)
x=x*(1+0.15*np.random.normal(size=x.size))

#set the measured size distribution parameters
#log-normal distribution
stddev = s
mean = m

if dist=='lognorm':
  distf=stats.lognorm(stddev,scale=mean)
elif dist=='normal':
  distf=stats.norm(scale=stddev*mean,loc=mean)

print(dist, 'StdDev=' , stddev , 'Mean=' , mean)

#%%

#start the iteration process
print('\tIter\tGOAL\t\tPOSITV\tSMOOTH\tCHISQR\t\t%PEN\n')
N_iter=1
fig=plt.figure()
procnow()
fig.set_size_inches(8,3)

if mode == 'slow':
  maxiter=None
elif mode == 'medium':
  maxiter=100
elif mode == 'fast':
  maxiter=50
else:
  maxiter=None

temp1=R_space
temp2=R_trim

res = minimize(goal, x, args=(pdf_sync,distf,R_trim), method=method, callback=printnow, options={'disp': True,'maxiter':maxiter,'return_all':True})

#%%
#Post-optimization data ploting/processing

#set the start point and end point to 0
res.x[0]=0
res.x[-1]=0

#plot the sharpened pdf 
fig1=plt.figure()
plt.plot(R_space,res.x/res.x.max(),'o') 

if fend == 'pickle':
  plt.plot(np.linspace(0,1,pdf.size), pdf/pdf.max(),':')
plt.xlabel(r'$r/D_{max}$')
plt.ylabel(r'$PDDF(r)$')
procnow()

fig2=plt.figure()
pdf_fit=synthetic(res.x,distf,R_trim)
pdf_sync_intp=np.interp(np.linspace(0,1,pdf_fit.size),np.linspace(0,1,pdf_sync.size),pdf_sync)
plt.plot(R_trim,pdf_sync_intp/pdf_sync_intp.max(),':')
plt.plot(R_trim,pdf_fit/pdf_fit.max(),'o')
plt.xlabel(r'$r$/$\AA$')
plt.ylabel(r'$PDDF(r)$')
procnow()

print('goal=')
print(goal(res.x,pdf_sync,distf,R_trim))

print('rmse=')
print(rmse(pdf_sync,synthetic(res.x,distf,R_trim)))

print('POSITV=')
print(POSITV(res.x))

print('SMOOTH=')
print(SMOOTH(res.x))

#%%

import pdf2iq

temp=res.x/res.x.max()
q,iq1=pdf2iq.pdf2iq(R_space,temp[:],mean=mean,q_min=0.001,q_max=1,q_size=2000,r_size=2000,thresold=0.001,base=0.001)


#%%
#data output

if save:
  if output_fname == None:
    now = datetime.now()
    output_fname = now.strftime("_%Y%m%d_%H%M%S")
    output_fname = (input_fname+output_fname)
    fname=(output_fname+'_log.pickle')
  with open(fname, 'wb') as f:
      pickle.dump(res, f)
        
  fname=(output_fname+'_dsmr.out')
  SAStool.printpddf(fname,q,iq1,R_space*m*2,res.x)
  
