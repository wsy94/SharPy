# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 16:33:02 2021

@author: dimo1
"""
from matplotlib import pyplot as plt

def procplotnow():
    ax=plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    fig.set_dpi(300)
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
    #plt.savefig('figure1.png',dpi=300)