# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:39:42 2021

@author: dimo1
"""

#read and print pddf data from gnom.out files

import numpy as np 
#import re
from datetime import datetime

#%%
def readpddf(fname):
    f=open(fname,"r")
    is_pddf_data=0
    #text=f.readlines()
    while is_pddf_data==0:
      if f.readline().find('P(R)')!=-1:
          data = np.genfromtxt(f, skip_header=1,skip_footer=2)
          is_pddf_data=1
    f.close()
    return data
    f.close()

#%%
def printpddf(fname,q,iqs,r,pddf):
    header='\n           ####      G N O M             Version 5.0 (r13469)      ####\n'
    now = datetime.now()
    date_time = now.ctime()
    try:
        rg,i0=quickrg(q,iqs)  
    except:
        rg=0
        i0=0
    
    #fname=('AuShell.out')
    f=open(fname,"x")
    f.writelines(header)
    f.write('\t\t\t'+date_time+'\n')
    
    f.write('\n           ####      Configuration                                 ####\n\n')
    f.write('  System Type:                   arbitrary monodisperse (job = 0)\n')     
    f.write('  System Type:                   arbitrary monodisperse (job = 0)\n')   
    f.write('  Minimum characteristic size:\t%.4f\n' % (r.min()))         
    f.write('  Maximum characteristic size:\t%.4f\n' % (r.max())) 
    
    f.write('\n            ####      Results                                       ####\n\n')
    f.write('  Angular range:                       %.4f to       %.4f\n' %(q.min(),q.max()))
    f.write('  Reciprocal space Rg:             %.4E\n' % (rg))
    f.write('  Reciprocal space I(0):           %.4E\n\n' % (i0))

    f.write('  Real space range:                    %.4f to     %.4f\n' % (r.min(),r.max()))
    f.write('  Real space Rg:                   %.4E\n' % (rg))
    f.write('  Real space I(0):                 %.4E\n\n' % (i0))
            
    f.write('\n           ####      Experimental Data and Fit                     ####\n\n') 
    f.write('      S          J EXP       ERROR       J REG       I REG\n\n')        
    iqs_err=np.gradient(iqs)
    for i in range(0,iqs.size):
        f.write("  %.6E\t%.6E\t%.6E\t%.6E\t%.6E\n" % (q[i], iqs[i], abs(iqs_err[i]), 0.0, 0.0 ) )
    
    f.write('\n    ####      Real Space Data                               ####\n\n')
    f.write('\n Distance distribution  function of particle  \n\n')
    f.write('\n       R          P(R)      ERROR\n\n')
    
    pddf_err=np.gradient(pddf)
    for i in range(0,pddf.size):
        f.write("  %.4E\t%.4E\t%.4E\n" % (r[i], pddf[i], abs(pddf_err[i])) )
    
    return fname
    f.close()
    

#%%
def quickrg(q,iqs,plot=False):
    qmax=q.max()
    qmin=q.min()
    fit_range = (q>=qmin)*(q<=qmax)
    Rg=1.3/qmax
    is_guinuir=0
    while is_guinuir==0:
        Rg2,iq0=np.polyfit(q[fit_range]**2,np.log(iqs[fit_range]),1)
        fit=np.polyval([Rg2,iq0],q[fit_range]**2)
        Rg=np.sqrt(abs(Rg2)*3)
        p=np.corrcoef(np.log(iqs[fit_range]),fit)
        qmax_new=1.3/Rg
        if p[0,1]>0.999 and Rg*qmax<1.33 and Rg*qmin<0.65:
            is_guinuir=1
        else:
                if qmax_new==qmax:
                    qmin=q[np.where(q==qmin)[0]+1]
        if is_guinuir == 0:
            qmax=qmax_new
            fit_range=  (q>=qmin)*(q<=qmax)
            
    if plot==True:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(q[fit_range]**2,np.log(iqs[fit_range]),'o')
        plt.plot(q[fit_range]**2,fit,':')
        plt.show()
    return Rg,np.exp(iq0)

#%%