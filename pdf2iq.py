# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:20:53 2021

@author: dimo1
"""

import numpy as np
from SAStool import quickrg
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
pi=np.pi


def pdf2iq(pdf_x, pdf_y, *args, r_size=None, mean=None, thresold=None, q_size=None, q_min=None, q_max=None, q_scale=None, draw=True, base=0, **kwargs):
    
    #default configs
    
    mean = 0.5 if mean is None else mean
    r_size = pdf_x.size if r_size is None else r_size
    q_size = 300 if q_size is None else q_size
    q_min = 0.5/pdf_x.max()/(2*mean) if q_min is None else q_min
    q_max = 100*q_min if q_max is None else q_max
    q_scale = 'linear' if q_scale is None else q_scale
    thresold = None if thresold is None else thresold
    base = 0 if base is None else base
    
    #determine d_max
    d_max=pdf_x.max()

    if mean != None:
        d_max=d_max*mean*2
        pdf_x=pdf_x/pdf_x.max()*d_max
        
    d_max=pdf_x[np.max(np.where((pdf_y[pdf_y>=thresold*pdf_y.max()])))]
    print('dmax=',d_max)

    #draw figures
    if draw==True:
        plt.figure()
        plt.subplot(221)
        plt.plot(pdf_x,pdf_y,'o')
    
    #pre——process the input pdf data
    
    n_init=3
    nn_init=np.int((n_init-1)/pdf_x.size*r_size)
    b=d_max
    
    def fit_init(x, a):
        return a * ((x)**5 - 3 * (x)**3 + 2 * (x)**2)
      
    fit_init_param = curve_fit(fit_init, (pdf_x[0:n_init]/b), pdf_y[0:n_init])
    a=fit_init_param[0]
    
    pdf_y=np.copy(pdf_y)
    pdf_y[0:n_init]=a*((pdf_x[0:n_init]/b)**5-3*(pdf_x[0:n_init]/b)**3+2*(pdf_x[0:n_init]/b)**2)
    
    # f=interp1d(pdf_x,pdf_y,kind='cubic')
    f=interp1d(pdf_x,pdf_y,kind='cubic')
    pddf_x=np.linspace(0,pdf_x.max(),r_size)
    pddf_y=f(pddf_x)
    
    pddf_y[0:nn_init]=a*((pddf_x[0:nn_init]/b)**5-3*(pddf_x[0:nn_init]/b)**3+2*(pddf_x[0:nn_init]/b)**2)
    
    if (thresold != None) and (thresold != 0):
        mask=pddf_y>=thresold*pddf_y.max()
        mask[0:pddf_y.argmax()]=True
        mask[np.min(np.where(mask==False)):-1]=False
        pddf_x=pddf_x[mask]
        pddf_y=pddf_y[mask]
        
    if draw==True:
        plt.plot(pddf_x,pddf_y,'-')
        plt.xlabel(r'$r$/$\AA$')
        plt.ylabel(r'$PDDF(r)$')

    r_space=pddf_x
    
    
    #create q space
    if q_scale=='log':
        q_space=np.logspace(np.log10(q_min),np.log10(q_max),q_size)
    elif q_scale=='linear':
        q_space=np.linspace(q_min,q_max,q_size)
    elif q_scale=='given':
        q_space=args
        
    iqs=np.zeros(q_size)
             
    for q in range(0,q_space.size-1):
        for r in range(0,r_space.size-1):
            dr=r_space[r+1]-r_space[r]
            pr=(pddf_y[r]+pddf_y[r+1])/2
#            pr=pddf_y[r]
            if pr<0:
                pr=0
            qr=(q_space[q]*r_space[r]+q_space[q+1]*r_space[r+1])/2
#            qr=(q_space[q]*r_space[r])/1
            sin_qr=np.sin(qr)
            iqs[q]=iqs[q]+pr*sin_qr*(1/qr)*dr
    
    #Rg,iqs[0] = quickrg(q_space[1:-1],iqs[1:-1])
    q_space[0] = 0
    iqs[-1]=iqs[-2]*(q_space[-2]/q_space[-1])**2
    
    iqs=iqs+base
    
#    Srq=np.zeros([r_space.size, q_space.size])
#    for q in range(0,q_space.size):
#        for r in range(1,r_space.size):
#            dr=r_space[r]-r_space[r-1]
#            qr=(q_space[q-1]*r_space[r-1]+q_space[q]*r_space[r])/2
#            #qr=(q_space[q]*r_space[r])/2
#            sin_qr=np.sin(qr)
#            Srq[r,q]=sin_qr*(1/qr)*dr
#            
#    iqs=np.zeros(Srq.shape[1])
#    for q in range(1,Srq.shape[1]):
#        for r in range(1,Srq.shape[0]):
#            pr=pddf_y[r]/2
#            iqs[q]=iqs[q]+pr*Srq[r,q]
    
    if draw==True:
        plt.subplot(222)
        plt.loglog(q_space,iqs)
        plt.xlabel(r'$q$/$\AA$^-1')
        plt.ylabel(r'Intensity (a. u.)')
        
        plt.subplot(223)
        plt.plot(q_space**2,np.log10(iqs))
        plt.xlabel(r'$q^2$/$\AA$^-2')
        plt.ylabel(r'log(I)')
#        plt.title('Rg=%.2f$\AA$'% (Rg) )
        #print('Rg=%.2f$\AA$'% (Rg) )
        
        plt.subplot(224)
        plt.plot(q_space,q_space**4*iqs)
        plt.xlabel(r'$q$/$\AA$^-1')
        plt.ylabel(r'q^4*I')
        
        plt.tight_layout()
        
    
    return q_space, iqs
  
  

def Srq(r_space, *args, q_min=None, q_max=None, q_size=None, q_scale=None, **kwargs):
    
    #default configs
    
    q_size = 100 if q_size is None else q_size
    q_min = 1/r_space.max() if q_min is None else q_min
    q_max = 50*q_min if q_max is None else q_max
    q_scale = 'log' if q_scale is None else q_scale
    
    if q_scale=='log':
        q_space=np.logspace(np.log10(q_min),np.log10(q_max),q_size)
    elif q_scale=='linear':
        q_space=np.linspace(q_min,q_max,q_size)
    elif q_scale=='given':
        q_space=args
        
    Srq=np.zeros([r_space.size, q_size])
    Srq[0,:]=q_space
    for q in range(0,q_space.size):
        for r in range(1,r_space.size):
            dr=r_space[r]-r_space[r-1]
            qr=(q_space[q-1]*r_space[r-1]+q_space[q]*r_space[r])/2
            #qr=(q_space[q]*r_space[r])/2
            sin_qr=np.sin(qr)
            Srq[r,q]=sin_qr*(1/qr)*dr
#        Srq[-1,q]=np.sin(q_space[q]*r_space[-1])*(1/q_space[q]*r_space[-1])*(r_space[-1]-r_space[-2])
        
    return Srq

def quickiq(pr,Srq):
    iqs=np.zeros(Srq.shape[1])
    for q in range(1,Srq.shape[1]):
        for r in range(1,Srq.shape[0]):
            iqs[q]=iqs[q]+pr[r]*Srq[r,q]
    
    return iqs
    
