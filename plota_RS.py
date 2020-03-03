import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pymc3 as pm
import scipy
from scipy import optimize
from scipy import stats
import matplotlib.ticker
from scipy.integrate import quad
import math
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import astropy.units as u
import csv
import astropy.constants as ctes
import matplotlib.cm as cm
import os
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy import optimize

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FUNÇÕES
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def le_file(nomes,file_in):
  PATH='/home/natalia/Dados/filamentos/SCMS'
  array_out=[]
  for n in range(len(nomes)):
    item=nomes[n]
    dir=PATH+'/'+item+'/'+file_in
    aux_df = pd.read_csv(dir)
    # aux_df=aux_df.drop(aux_df.columns[0], axis=1)
    array_out.append(aux_df)
  return array_out
#*****************

#=================================================================
# LE ARQUIVOS
#===============================================================
f_agl='/home/natalia/Dados/Aglomerados/Xdata_total.csv'
f_agl=pd.read_csv(f_agl, delimiter=',')
f_agl=f_agl.sort_values(by=[' redshift'])
f_agl=f_agl.reset_index(drop=True)
nomes=f_agl['Cluster']


dfx=f_agl
colors = cm.tab20(np.linspace(0, 1, len(dfx.index.values)))


PATH='/home/natalia/Dados/filamentos/SCMS'

#VALORES AJUSTE RS
file_in='RS_fit_Total.csv'
RSfit = le_file(nomes,file_in)

#GALAXIAS DO CLUSTER
file_in='GalsInCl_Clean_final.csv'
GalsInCl = le_file(nomes, file_in)

#GALAXIAS DO CLUSTER DIRTY
file_in='GalsInCluster_dirtyWprob.csv'
GalsInCl_Dirty = le_file(nomes, file_in)

#GALAXIAS DO FILAMENTO
file_in='GalsInFil_Grad_Clean.csv'
GalsInFil = le_file(nomes, file_in)

#GALAXIAS DO FILAMENTO DIRTY
file_in='GalsInFil_noCl_noColorCut.csv'
GalsInFil_Dirty = le_file(nomes, file_in)


#=================================================================
# PLOTA RS CLUSTER
#===============================================================

limMrsup=-24.5
limMrinf=-18


fig,axs=plt.subplots(nrows=3,ncols=5,sharey=True,sharex=True)
for n in range(len(nomes)):
  Mg=GalsInCl[n]['Mg']
  Mr=GalsInCl[n]['Mr']
  gr=GalsInCl[n]['gr']
  MgD=GalsInCl_Dirty[n]['Mg']
  MrD=GalsInCl_Dirty[n]['Mr']
  grD=GalsInCl_Dirty[n]['gr']
  rs_ang=RSfit[n]['Ang_RS_cluster']
  rs_lin=RSfit[n]['ZeroPoint_RS_cluster']

  x1 = [ limMrsup, limMrinf]
  y1 = [ rs_ang*limMrsup + rs_lin,  rs_ang*limMrinf + rs_lin ]
  yh = [rs_ang*limMrsup + rs_lin +0.15 ,  rs_ang*limMrinf + rs_lin +0.15]


  #dirty
  axs.flat[n].scatter(MrD,grD, s=10, color=colors[0], marker='o')
  #clean
  axs.flat[n].scatter(Mr,gr, s=10, color=colors[8], marker='o')
  #RS
  axs.flat[n].plot(x1, yh, color='grey', linestyle=':')
  axs.flat[n].plot(x1, y1, color='grey', linestyle='-')
  #MR_lims
  axs.flat[n].axvline(limMrinf,color='grey', linestyle=':')
  axs.flat[n].axvline(limMrsup,color='grey', linestyle=':')

  axs.flat[n].set_xlim(-25,-17.5)
  axs.flat[n].set_ylim(0,1.25)


  axs.flat[n].tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
  tit=nomes[n]+' - z= %.3f' % f_agl[' redshift'][n]
  axs.flat[n].set_title(tit, pad=3., fontsize=8, fontweight='bold')

xlab=r'$M_r$'
ylab=r'$g-r$'
fig.text(0.5, 0.05, xlab, va='center')
fig.text(0.07, 0.5, ylab, va='center', rotation='vertical')
fig.set_size_inches(12.5,8.5)

plt.savefig('Cluster_RSfit.png')
plt.close()
#=================================================================
# PLOTA RS FILAMENTO
#===============================================================

limMrsup=-24.5
limMrinf=-18


fig,axs=plt.subplots(nrows=3,ncols=5,sharey=True,sharex=True)
for n in range(len(nomes)):
  Mg=GalsInFil[n]['Mg']
  Mr=GalsInFil[n]['Mr']
  gr=GalsInFil[n]['gr']
  MgD=GalsInFil_Dirty[n]['Mg']
  MrD=GalsInFil_Dirty[n]['Mr']
  grD=GalsInFil_Dirty[n]['gr']
  rs_ang=RSfit[n]['Ang_RS_cluster']
  rs_lin=RSfit[n]['ZeroPoint_RS_cluster']

  x1 = [ limMrsup, limMrinf]
  y1 = [ rs_ang*limMrsup + rs_lin,  rs_ang*limMrinf + rs_lin ]
  yh = [rs_ang*limMrsup + rs_lin +0.15 ,  rs_ang*limMrinf + rs_lin +0.15]
  yl = [rs_ang*limMrsup + rs_lin -0.15 ,  rs_ang*limMrinf + rs_lin -0.15]


  #dirty
  axs.flat[n].scatter(MrD,grD, s=10, color=colors[0], marker='o')
  #clean
  axs.flat[n].scatter(Mr,gr, s=10, color=colors[8], marker='o')
  #RS
  axs.flat[n].plot(x1, yh, color='grey', linestyle=':')
  axs.flat[n].plot(x1, y1, color='grey', linestyle='-')
  axs.flat[n].plot(x1, yl, color='grey', linestyle=':')
  #MR_lims
  axs.flat[n].axvline(limMrinf,color='grey', linestyle=':')
  axs.flat[n].axvline(limMrsup,color='grey', linestyle=':')

  axs.flat[n].set_xlim(-25,-17.5)
  axs.flat[n].set_ylim(-0.5,1.25)


  axs.flat[n].tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
  tit=nomes[n]+' - z= %.3f' % f_agl[' redshift'][n]
  axs.flat[n].set_title(tit, pad=3., fontsize=8, fontweight='bold')

xlab=r'$M_r$'
ylab=r'$g-r$'
fig.text(0.5, 0.05, xlab, va='center')
fig.text(0.07, 0.5, ylab, va='center', rotation='vertical')
fig.set_size_inches(12.5,8.5)

plt.savefig('Fil_RSfit.png')
plt.close()

