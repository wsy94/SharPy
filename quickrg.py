# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:40:27 2021

@author: dimo1
"""

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
    if p[0,1]>0.999 and Rg*qmax<1.33 and Rg*qmin<0.65:
        is_guinuir=1
    else:
            if qmax_new==q_max:
                qmin=q[np.where(q==qmin)[0]+1]
    if is_guinuir == 0:
        qmax_new=1.3/Rg
        qmax=qmax_new
        fit_range=  (q>=qmin)*(q<=qmax)
    
plt.plot(q[fit_range]**2,np.log(iqs[fit_range]),'o')
plt.plot(q[fit_range]**2,fit,':')
procnow()
plt.show()