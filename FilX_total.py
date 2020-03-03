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
from scipy.odr import *
from scipy.optimize import least_squares
from M200 import M200

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

def plota_f1(dfx, dfy1, colx, coly, colxErr, colyErr,x_label, y_label, outname):
  idx_x=dfx.index.values

  erx=[]
  ery=[]
  if (colxErr == 0):
    for n in range(len(idx_x)):
      item = idx_x[n]
      erx.append(0)
  else:
    if colx == ' kTm': 
      for n in range(len(idx_x)):
        item = idx_x[n]
        erx.append(dfx.loc[item][colxErr] - dfx.loc[item][colx])
    else:
      for n in range(len(idx_x)):
        item = idx_x[n]
        erx.append(dfx.loc[item][colxErr])

  if (colyErr == 0):
    for n in range(len(idx_x)):
      item = idx_x[n]
      ery.append(0)
  else:
    for n in range(len(idx_x)):
      item = idx_x[n]
      ery.append(dfy1.loc[item][colxErr])

  f, (ax1)=plt.subplots(ncols=1, nrows=1, sharex=True)
  colors = cm.tab20(np.linspace(0, 1, len(idx_x)))

  legs=[]
  for n in range(len(idx_x)):
    item = idx_x[n]
    if (dfx.loc[item][' CC'] == 'NCC') :
      marca='o'
    if (dfx.loc[item][' CC'] == 'CC') :
      marca='d'
    # ax1.scatter(dfx.loc[item][colx], dfy1.loc[item][coly], color=colors[n], label=item)
    ax1.errorbar(dfx.loc[item][colx], dfy1.loc[item][coly], xerr=erx[n], yerr=ery[n], marker=marca, capsize=5, mec='k', mfc=colors[n], ms=7, elinewidth=1, ecolor=colors[n], label=dfx.loc[item]['Cluster'])
    legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[n], mec='k', markersize=9, alpha=0.7, label=dfx.loc[item]['Cluster']))
  legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='NCC'))
  # legs.append(Line2D([0], [0], marker='s', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='Merger'))
  legs.append(Line2D([0], [0], marker='d', color='w',markerfacecolor="k", markersize=7, alpha=0.7, label='CC'))
  ax1.set_xlabel(x_label)
  ax1.set_ylabel(y_label)

  ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  ax1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

  f.subplots_adjust(hspace=0, right=0.7)
  plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
  plt.legend(handles=legs, loc='center left', bbox_to_anchor=(1,0.5),
          ncol=1, fancybox=True, shadow=True, fontsize = 'x-small')
  outn=outname+'_CC.png'
  plt.savefig(outn)
  plt.close()

  f, (ax1)=plt.subplots(ncols=1, nrows=1, sharex=True)
  colors = cm.tab20(np.linspace(0, 1, len(idx_x)))

  legs=[]
  for n in range(len(idx_x)):
    item = idx_x[n]
    if (dfx.loc[item][' Status'] == 'merge') :
      marca='s'
    if (dfx.loc[item][' Status'] == 'relax') :
      marca='X'
    if (dfx.loc[item][' Status'] == 'interm') :
      marca='v'
    # ax1.scatter(dfx.loc[item][colx], dfy1.loc[item][coly], color=colors[n], label=item)
    ax1.errorbar(dfx.loc[item][colx], dfy1.loc[item][coly], xerr=erx[n], yerr=ery[n], marker=marca, capsize=5, mec='k', mfc=colors[n], ms=7, elinewidth=1, ecolor=colors[n], label=dfx.loc[item]['Cluster'])
    legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[n], mec='k', markersize=9, alpha=0.7, label=dfx.loc[item]['Cluster']))
  legs.append(Line2D([0], [0], marker='s', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='Merging'))
  legs.append(Line2D([0], [0], marker='X', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='Relaxed'))
  legs.append(Line2D([0], [0], marker='v', color='w',markerfacecolor="k", markersize=7, alpha=0.7, label='Intermediate'))
  ax1.set_xlabel(x_label)
  ax1.set_ylabel(y_label)

  ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  ax1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

  f.subplots_adjust(hspace=0, right=0.7)
  plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
  plt.legend(handles=legs, loc='center left', bbox_to_anchor=(1,0.5),
          ncol=1, fancybox=True, shadow=True, fontsize = 'x-small')
  outn=outname+'_status.png'
  plt.savefig(outn)
  plt.close()

  return
#*****************

def plota_CC(cx, cy, cxErr, cyErr, dt_aux, x_label, y_label):
  #dt_aux eh a tabela que contem NOMES, CC, STATUS
  nomes=dt_aux['Cluster'].values
  legs=[]
  for n in range(len(nomes)):
    # yy=frac_red[n][coly]/frac_color[' RF_tot_CMP'][n]
    item=nomes[n]
    if (dt_aux[' CC'][n] == 'NCC') :
      marca='o'
    elif (dt_aux[' CC'][n] == 'CC') :
      marca='d'
    else :
        marca=''

    plt.errorbar(cx[n], cy[n], xerr=cxErr[n], yerr=cyErr[n], marker=marca, capsize=2, mec='k', mfc=colors[n], \
  ms=6, elinewidth=1, ecolor=colors[n])

    legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[n], mec='k', markersize=9, alpha=0.7, label=item))
  legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='NCC'))
  legs.append(Line2D([0], [0], marker='d', color='w',markerfacecolor="k", markersize=7, alpha=0.7, label='CC'))
  plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
  plt.subplots_adjust(hspace=0, right=0.7)
  plt.legend(handles=legs, loc='center left', bbox_to_anchor=(1,0.5),
            ncol=1, fancybox=True, shadow=True, fontsize = 'x-small')

   
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.show()

  return



#*****************

def plota_status(cx, cy, cxErr, cyErr, dt_aux, x_label, y_label):
  #dt_aux eh a tabela que contem NOMES, CC, STATUS
  nomes=dt_aux['Cluster'].values
  legs=[]
  for n in range(len(nomes)):
    # yy=frac_red[n][coly]/frac_color[' RF_tot_CMP'][n]
    item=nomes[n]
    if (dt_aux[' Status'][n] == 'merge') :
      marca='s'
    elif (dt_aux[' Status'][n] == 'relax') :
      marca='X'
    elif (dt_aux[' Status'][n] == 'interm') :
        marca='v'
    else :
        marca=''

    plt.errorbar(cx[n], cy[n], xerr=cxErr[n], yerr=cyErr[n], marker=marca, capsize=2, mec='k', mfc=colors[n], \
  ms=6, elinewidth=1, ecolor=colors[n])

    legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[n], mec='k', markersize=9, alpha=0.7, label=item))
  legs.append(Line2D([0], [0], marker='s', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='Merging'))
  legs.append(Line2D([0], [0], marker='X', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='Relaxed'))
  legs.append(Line2D([0], [0], marker='v', color='w',markerfacecolor="k", markersize=7, alpha=0.7, label='Intermediate'))
  plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
  plt.subplots_adjust(hspace=0, right=0.7)
  plt.legend(handles=legs, loc='center left', bbox_to_anchor=(1,0.5),
            ncol=1, fancybox=True, shadow=True, fontsize = 'x-small')

  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.show()

  return

#*****************

def plota_CC_HD(cx, cy, cxErr, cyErr, dt_aux, x_label, y_label):
  #dt_aux eh a tabela que contem NOMES, CC, STATUS
  nomes=dt_aux['Cluster'].values
  legs=[]
  for n in range(len(nomes)):
    # yy=frac_red[n][coly]/frac_color[' RF_tot_CMP'][n]
    item=nomes[n]
    if (dt_aux[' CC'][n] == 'NCC') :
      marca='o'
    elif (dt_aux[' CC'][n] == 'CC') :
      marca='d'
    else :
        marca=''
    if dt_aux['HighDens'][n] == 0:
      alp = 0.6
      meww=2.
    else:
      alp=1
      meww=1


    plt.errorbar(cx[n], cy[n], xerr=cxErr[n], yerr=cyErr[n], marker=marca, capsize=2, mec='k', mfc=colors[n], \
  ms=6, elinewidth=1, ecolor=colors[n],alpha=alp,mew=meww)

    legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[n], mec='k', markersize=9, alpha=0.7, label=item))
  legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='NCC'))
  legs.append(Line2D([0], [0], marker='d', color='w',markerfacecolor="k", markersize=7, alpha=0.7, label='CC'))
  plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
  plt.subplots_adjust(hspace=0, right=0.7)
  plt.legend(handles=legs, loc='center left', bbox_to_anchor=(1,0.5),
            ncol=1, fancybox=True, shadow=True, fontsize = 'x-small')

   
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.show()

  return
#*****************

def plota_CC_dist(cx, cy, cxErr, cyErr, dt_aux, x_label, y_label):
  #dt_aux eh a tabela que contem NOMES, CC, STATUS
  nomes=dt_aux['Cluster'].values
  legs=[]
  for n in range(len(nomes)):
    # yy=frac_red[n][coly]/frac_color[' RF_tot_CMP'][n]
    item=nomes[n]
    if (dt_aux[' CC'][n] == 'NCC') :
      marca='o'
    if (dt_aux[' CC'][n] == 'CC') :
      marca='d'

    plt.errorbar(cx, cy[n], yerr=cyErr[n], marker=marca, color=colors[n],capsize=5, mec='k', mfc=colors[n], \
  ms=6)

    legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[n], mec='k', markersize=9, alpha=0.7, label=item))
  legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='NCC'))
  legs.append(Line2D([0], [0], marker='d', color='w',markerfacecolor="k", markersize=7, alpha=0.7, label='CC'))
  plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)

  plt.subplots_adjust(hspace=0, right=0.7)
  plt.legend(handles=legs, loc='center left', bbox_to_anchor=(1,0.5),
            ncol=1, fancybox=True, shadow=True, fontsize = 'x-small')

   
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.show()

  return
#*****************

def plota_status_dist(cx, cy, cxErr, cyErr, dt_aux, x_label, y_label):
  #dt_aux eh a tabela que contem NOMES, CC, STATUS
  nomes=dt_aux['Cluster'].values
  legs=[]
  for n in range(len(nomes)):
    # yy=frac_red[n][coly]/frac_color[' RF_tot_CMP'][n]
    item=nomes[n]
    if (dt_aux[' Status'][n] == 'merge') :
      marca='s'
    if (dt_aux[' Status'][n] == 'relax') :
      marca='X'
    if (dt_aux[' Status'][n] == 'interm') :
        marca='v'  

    plt.errorbar(cx, cy[n], yerr=cyErr[n], marker=marca, color=colors[n],capsize=5, mec='k', mfc=colors[n], \
  ms=6)

    legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[n], mec='k', markersize=9, alpha=0.7, label=item))
  legs.append(Line2D([0], [0], marker='s', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='Merging'))
  legs.append(Line2D([0], [0], marker='X', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='Relaxed'))
  legs.append(Line2D([0], [0], marker='v', color='w',markerfacecolor="k", markersize=7, alpha=0.7, label='Intermediate'))

  plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
  plt.subplots_adjust(hspace=0, right=0.7)
  plt.legend(handles=legs, loc='center left', bbox_to_anchor=(1,0.5),
            ncol=1, fancybox=True, shadow=True, fontsize = 'x-small')

   
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.show()

  return
#*****************

def media_cor(mg1,mg2):
  val=(sum(mg1-mg2)/len(mg1))
  err=np.std(mg1-mg2)/np.sqrt(len(mg1))
  return val, err

def media_pond(mg1,mg2,P):
  x1=sum(P*(mg1-mg2))
  x2=sum(P)
  val=x1/x2
  err=1/np.sqrt(x2)
  return val, err
#*****************
def plota_dist2fil_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey):
  cy=[]
  cyErr=[]
  dists=[0,0.5,1.,1.5]
  for i in range(len(dists)-1):
    auxi=[]
    auxiEr=[]
    for n in range(len(nomes)):
      aux1=filey[n].loc[(dists[i] < filey[n][nomcy])]
      aux2= aux1.loc[aux1[nomcy]<= dists[i+1]]
      auxi.append(np.mean(aux2.gi))
      auxiEr.append(np.std(aux2.gi)/np.sqrt(len(aux2.gi)))  
    cy.append(auxi)
    cyErr.append(auxiEr)

  fig, ax=plt.subplots(nrows=1,ncols=3, sharex=True, sharey=True)
  xs=ax.flat
  for n in range(len(dists)-1):
    axis=xs[n]
    axis.errorbar(cx, cy[n], xerr=cxErr, yerr=cyErr[n], marker='o', capsize=2, mec='k', mfc=colors[n], \
  ms=6, elinewidth=1, ecolor=colors[n],linestyle=' ')

    tit=str(dists[n]) + r'$<$ Dist2Fil $<=$' + str(dists[n+1]) + 'Mpc'
    axis.set_title(tit, fontsize=8, fontweight='bold')
    axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)

  fig.text(0.5, 0.05, x_label, va='center')
  fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')
  fig.set_size_inches(12.5,4.5)
  plt.show()
  return

def plota_dist2cluster_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey):
  cy=[]
  cyErr=[]
  # dists=[0.,1.,2.,3.,4.,5.,6]
  dists=[1.5,2.5,3.5,4.5,5.5]
  for i in range(len(dists)-1):
    auxi=[]
    auxiEr=[]
    for n in range(len(nomes)):
      aux1=filey[n].loc[(dists[i] < filey[n][nomcy])]
      aux2= aux1.loc[aux1[nomcy]<= dists[i+1]]
      auxi.append(np.mean(aux2.gi))
      auxiEr.append(np.std(aux2.gi)/np.sqrt(len(aux2.gi)))
    cy.append(auxi)
    cyErr.append(auxiEr)

  fig, ax=plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True)
  xs=ax.flat
  for n in range(len(dists)-1):
    axis=xs[n]
    axis.errorbar(cx, cy[n], xerr=cxErr, yerr=cyErr[n], marker='o', capsize=2, mec='k', mfc=colors[n], \
  ms=6, elinewidth=1, ecolor=colors[n],linestyle=' ')

    tit=str(dists[n]) + r'$<$ Dist2Agl $<=$' + str(dists[n+1]) + 'Mpc'
    axis.set_title(tit, fontsize=8, fontweight='bold')
    axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)


  fig.text(0.5, 0.05, x_label, va='center')
  fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')

  plt.show()
  return

def reta(x,a,b):
  return (a*x + b)

def reta2(p, x):
   m, c = p
   return m*x + c

def curva2(p, x):
   m, c = p
   return (m/x) + c

def fitReta(x,y,xerr,yerr):
  linear=Model(reta2)
  mydata = RealData(x, y, sx=xerr, sy=yerr)
  myodr = ODR(mydata, linear, beta0=[1.,0.])
  myoutput = myodr.run()
  myoutput.pprint()
  beta=myoutput.beta
  std=myoutput.sd_beta
  return beta,std

def fitcurva(x,y,xerr,yerr):
  linear=Model(curva2)
  mydata = RealData(x, y, sx=xerr, sy=yerr)
  myodr = ODR(mydata, linear, beta0=[1.,0.])
  myoutput = myodr.run()
  myoutput.pprint()
  beta=myoutput.beta
  std=myoutput.sd_beta
  return beta,std
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LẼ TODAS AS ENTRADAS DE INTERESSE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#RAIOS-X---------------------------------------------------
f_agl='/home/natalia/Dados/Aglomerados/Xdata_total.csv'
f_agl=pd.read_csv(f_agl, delimiter=',')
f_agl=f_agl.sort_values(by=[' redshift'])
f_agl=f_agl.reset_index(drop=True)
nomes=f_agl['Cluster']

#OPTICO----------------------------------------------------
PATH='/home/natalia/Dados/filamentos/SCMS'

#AGLOMERADOS
file_in='GalsInCl_Clean_final.csv'
GalsInCl = le_file(nomes, file_in)

#FILAMENTOS
#comprimento total
file_in='f1_comrpimento.txt'
ProprFil = le_file(nomes,file_in)

#Galaxias nos filamentos, com informação do gradiente
file_in='GalsInFil_Grad_Dirty.csv'
GalsInFil = le_file(nomes,file_in)
GalsInFil_clean = []
GalsInFil_clean02 = []
GalsInCl_drt=[]
for n in range(len(nomes)):
  aux = GalsInFil[n].loc[GalsInFil[n].ClusterMember == 0]
  aux2 = GalsInFil[n].loc[GalsInFil[n].ClusterMember != 1]
  aux3 = GalsInFil[n].loc[GalsInFil[n].ClusterMember != 0]
  GalsInFil_clean.append(aux)
  GalsInFil_clean02.append(aux2)
  GalsInCl_drt.append(aux3)

#Galaxias nos filamentos em FATIAS
file_in='GalsInFil_Slice_Clean.csv'
GalsInFil_Slc = le_file(nomes,file_in)

#RED SEQUENCE
file_in='RS_fit_Total.csv'
RSfit = le_file(nomes,file_in)

#CONNECT 
file_in=PATH+'/Connect.csv'
Connect=pd.read_csv(file_in,delimiter=',') #já está ordenada por z
file_in=PATH+'/Connect_final.csv'
Connectf=pd.read_csv(file_in,delimiter=',') #já está ordenada por z

#DENSIDADE RELATIVA EM FATIAS
file_in='/GalsInFil_Grad_DensiRel.csv'
GalsInFil_GradDensi = le_file(nomes,file_in) #já está ordenada por z

dfx=f_agl
idx_x=dfx.index.values
colors = cm.tab20(np.linspace(0, 1, len(dfx.index.values)))


f_agl['HighDens'] = 1
f_agl['HighDens'][0] = 0
f_agl['HighDens'][13] = 0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# COR AGL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cx=f_agl[' redshift'].values
cxErr=[0]*len(cx)
cy=[]
cyErr=[]
cx2=[]
cy2=[]
cy2Err=[]
for n in range(len(nomes)):
  # cy.append(np.mean(GalsInCl_drt[n]['gi']))
  # cyErr.append(np.std(GalsInCl_drt[n]['gi'])/np.sqrt(len(GalsInCl_drt[n]['gi'])))
  cy.append(np.mean(GalsInCl[n]['gi']))
  cyErr.append(np.std(GalsInCl[n]['gi'])/np.sqrt(len(GalsInCl[n]['gi'])))
  if f_agl['HighDens'][n]==1:
    cx2.append(cx[n])
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
cx2Err=[0]*len(cx2)

x_label=r'$z_{agl}$'
y_label=r'$(g-i)_{agl}$'

plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
# plota_status(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
xfake=np.linspace(min(cx)-0.1, max(cx)+0.1,100)
#reta - tudo
# cx=np.array(cx).ravel()
# cy=np.array(cy).ravel()
# pars=RobustLinear(cx,cy)
# yfit=reta(pars[4],pars[0],pars[1])
# yfitS=reta(pars[4],pars[0]+np.sqrt(pars[3]),pars[1]+np.sqrt(pars[3]))
# yfitI=reta(pars[4],pars[0]-np.sqrt(pars[3]),pars[1]-np.sqrt(pars[3]))
pars,errs=fitReta(cx,cy,cxErr)
yfit=reta(xfake,pars[0],pars[1])
yfitS=reta(xfake,pars[0]+np.sqrt(errs[0][0]),pars[1]+np.sqrt(errs[1][1]))
yfitI=reta(xfake,pars[0]-np.sqrt(errs[0][0]),pars[1]-np.sqrt(errs[1][1]))
plt.plot(xfake,yfit,color='grey')
# plt.fill(
#     np.append(xfake, xfake[::-1]),
#     np.append(yfitI, yfitS[::-1]),
#     color='gray',alpha=0.5,edgecolor='w'
# )
fit_lab=r'Total: (%.2f $\pm$ %.2f)X + (%.2f $\pm$ %.2f)' % (pars[0],np.sqrt(errs[0][0]),pars[1],np.sqrt(errs[1][1]))
plt.text(0.15, 1.22, fit_lab, va='center',color='grey',\
  bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))
plt.text(0.15, 1.22, fit_lab, va='center',color='grey',\
  bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))

#reta - clean


cx2=np.array(cx2).ravel()
cy2=np.array(cy2).ravel()
# xfake=np.linspace(min(cx), max(cx),100)
# pars=RobustLinear(cx2,cy2)
# yfit=reta(pars[4],pars[0],pars[1])
# yfitS=reta(pars[4],pars[0]+np.sqrt(pars[3]),pars[1]+np.sqrt(pars[3]))
# yfitI=reta(pars[4],pars[0]-np.sqrt(pars[3]),pars[1]-np.sqrt(pars[3]))

# cx=np.array(cx[:-1]).ravel()
# cy=np.array(cy[:-1]).ravel()
pars,errs=optimize.curve_fit(reta,cx2,cy2)
yfit=reta(xfake,pars[0],pars[1])
yfitS=reta(xfake,pars[0]+np.sqrt(errs[0][0]),pars[1]+np.sqrt(errs[1][1]))
yfitI=reta(xfake,pars[0]-np.sqrt(errs[0][0]),pars[1]-np.sqrt(errs[1][1]))
plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='k',alpha=0.2,edgecolor='w'
)

fit_lab=r'Clean: (%.2f $\pm$ %.2f)X + (%.2f $\pm$ %.2f)' % (pars[0],np.sqrt(errs[0][0]),pars[1],np.sqrt(errs[1][1]))
plt.text(0.15, 0.98, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='k', boxstyle='round,pad=1'))


plt.xlim(0.135,0.355)
# plt.ylim(0.85,1.4)
plt.ylim(0.95,1.25)
# plt.title(r'$z \times(g-i)_{agl}$', fontsize=10)
outname='z_giCl_tot.png'
plt.savefig(outname)
plt.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# COR FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#COR MEDIA POR DISTANCIA AO CLS-----------------------------------
cx=[2.5,3.5,4.5]
# cx=[2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
cxErr=[0]*len(cx)
cy=[]
cyErr=[]
cy2=[]
cyErr2=[]
# dists=[2,3,4,5,6,7,8,9,10]
dists=[2,3,4,5]
ll=[]
ll_pond=[]
cyH=[]
cyHErr=[]
cyH2=[]
cyHErr2=[]


for i in range(len(dists)-1):
  auxi=[]
  auxiEr=[]
  auxiH=[]
  auxiHEr=[]
  auxll=[]
  aux_pond=14
  for n in range(len(nomes)):
    aux1=GalsInFil_clean02[n].loc[(dists[i] < GalsInFil_clean02[n].Dist2ClosCluster_Mpc)]
    aux2= aux1.loc[aux1.Dist2ClosCluster_Mpc<= dists[i+1]]
    auxll.append(len(aux2.gi))
    if len(aux2.gi) == 0:
      print('aa')
      aux_pond = aux_pond -1
    auxi.append(np.mean(aux2.gi))
    auxiEr.append(np.std(aux2.gi)/np.sqrt(len(aux2.gi)))
    auxii = [x for x in auxi if str(x) != 'nan']
    auxiiEr = [x for x in auxiEr if str(x) != 'nan']

    if f_agl['HighDens'][n] == 1:
      aux1=GalsInFil_clean02[n].loc[(dists[i] < GalsInFil_clean02[n].Dist2ClosCluster_Mpc)]
      aux2= aux1.loc[aux1.Dist2ClosCluster_Mpc<= dists[i+1]]
      auxiH.append(np.mean(aux2.gi))
      auxiHEr.append(np.std(aux2.gi)/np.sqrt(len(aux2.gi)))
      auxiiH = [x for x in auxi if str(x) != 'nan']
      auxiiHEr = [x for x in auxiEr if str(x) != 'nan']

  cy.append(np.mean(auxii))
  cyErr.append(np.std(auxii)/np.sqrt(len(auxi)))
  cy2.append(auxi)
  cyErr2.append(auxiEr)

  cyH.append(np.mean(auxiiH))
  cyHErr.append(np.std(auxiiH)/np.sqrt(len(auxiH)))
  cyH2.append(auxiH)
  cyHErr2.append(auxiHEr)

  ll.append(sum(auxll))
  ll_pond.append(sum(auxll)/aux_pond)



#plota media das cores pra todos os aglomerados em cada fatia
plt.errorbar(cx, cy, xerr=cxErr, yerr=cyErr, marker='o', capsize=2, mec='k', mfc=colors[0], \
ms=6, elinewidth=1, ecolor=colors[0],linestyle=' ')
plt.xlabel(r'D$_{agl}$ (Mpc)')
plt.ylabel(r'$(g-i)_{fil}$')
plt.tick_params(direction='in',top=True, right=True,labeltop=False, labelright=False)
# plt.ylim(0.85,1.4)
cxErr=[0.5]*3
linear=Model(reta2)
mydata = RealData(cx, cy, sy=cyErr,sx=cxErr)
myodr = ODR(mydata, linear, beta0=[1.,0.])
myoutput = myodr.run()
myoutput.pprint()
pars=myoutput.beta
errs=myoutput.sd_beta

xfake=np.linspace(min(cx)-1,max(cx)+1,100)
yfit=reta2(pars,xfake)
yfitS=reta2(pars+errs,xfake)
yfitI=reta2(pars-errs,xfake)

plt.plot(xfake,yfit,color='gray')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='dimgray',alpha=0.3,edgecolor='w'
)
plt.ylim(1.,1.15)
plt.xlim(2,5)

fit_lab=r'(%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(2.2, 0.98, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))


# plt.title(r'$(g-i)_{fil} \times$ Distance to closest cluster ', fontsize=10)
outname='Dist2Cl_giFil_fatiasMedia_5Fit.png'
plt.savefig(outname)
plt.close()

#plota media das cores POR AGLOMERADO em cada fatia
fig,ax = plt.subplots(nrows=7, ncols=2, sharex=True, sharey=True)
xs=ax.flat
for n in range(len(nomes)):
  aux=[]
  auxErr=[]
  for i in range(len(dists)-1):
    aux.append(cy2[i][n])
    auxErr.append(cyErr2[i][n])
    item = idx_x[n]
    if (dfx[' Status'][n] == 'merge') :
      marca='s'
    if (dfx[' Status'][n] == 'relax') :
      marca='X'
    if (dfx[' Status'][n] == 'interm') :
        marca='v'  
  axis=xs[n]
  axis.errorbar(cx, aux, xerr=cxErr, yerr=auxErr, marker=marca, capsize=2, mec='k', mfc=colors[n], \
ms=6, elinewidth=1, ecolor=colors[n],linestyle='-',color=colors[n])
  tit=nomes[n]
  axis.set_title(tit, fontsize=8, fontweight='bold', pad=-10)
  axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
ylab=r'$(g-i)_{fil}$'
xlab='Distance to closest cluster (Mpc)'
fig.text(0.4, 0.05, xlab, va='center')
fig.text(0.07, 0.5, ylab, va='center', rotation='vertical')
fig.set_size_inches(12.5, 8.5)
fig.set_ylim(0.85,1.4)
outname='Dist2Cl_giFil_fatiasIndividual.png'
plt.savefig(outname)
plt.close()


#COR MEDIA POR DISTANCIA AO FIL-----------------------------------
cx=[0.5,1.,1.5]
cxErr=[0]*len(cx)
cy=[]
cyErr=[]
cy2=[]
cyErr2=[]
dists=[0.,0.5,1.,1.5]
for i in range(len(dists)-1):
  auxi=[]
  auxiEr=[]
  for n in range(len(nomes)):
    aux1=GalsInFil_clean02[n].loc[(GalsInFil_clean02[n].Dist2Fil_Mpc > dists[i])]
    aux2= aux1.loc[aux1.Dist2Fil_Mpc<= dists[i+1]]
    auxi.append(np.mean(aux2.gi))
    auxiEr.append(np.std(aux2.gi)/np.sqrt(len(aux2.gi)))
    auxii = [x for x in auxi if str(x) != 'nan']
    auxiiEr = [x for x in auxiEr if str(x) != 'nan']
  cy.append(np.mean(auxii))
  cyErr.append(np.std(auxii)/np.sqrt(len(auxii)))
  cy2.append(auxi)
  cyErr2.append(auxiEr)

#plota media das cores pra todos os aglomerados em cada fatia
plt.errorbar(cx, cy, xerr=cxErr, yerr=cyErr, marker='o', capsize=2, mec='k', mfc=colors[0], \
ms=6, elinewidth=1, ecolor=colors[0],linestyle=' ')
plt.ylim(01.,1.15)
plt.xlabel(r'D$_{fil}$ (Mpc)')
plt.ylabel(r'$(g-i)_{fil}$')
plt.tick_params(direction='in',top=True, right=True,labeltop=False, labelright=False)

cxErr=[0.25]*3

linear=Model(reta2)
mydata = RealData(cx, cy, sx=cxErr, sy=cyErr)
myodr = ODR(mydata, linear, beta0=[1.,0.])
myoutput = myodr.run()
myoutput.pprint()
pars=myoutput.beta
errs=myoutput.sd_beta


xfake=np.linspace(min(cx)-1,max(cx)+1,100)
yfit=reta2(pars,xfake)
yfitS=reta2(pars+errs,xfake)
yfitI=reta2(pars-errs,xfake)

plt.plot(xfake,yfit,color='gray')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='dimgray',alpha=0.3,edgecolor='w'
)
plt.ylim(1.,1.15)
plt.xlim(0.25,1.75)



# fit_lab=r'(%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
fit_lab=r'(0.14 $\pm$ 4.9)10$^{-3}$X + (%.2f $\pm$ %.2f)' % (pars[1],errs[1])

plt.text(0.35, 1.015, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))

outname='Dist2Fil_giFil_fatiasMedia.png'
plt.savefig(outname)
plt.close()


#plota media das cores POR AGLOMERADO em cada fatia
fit_giDist2Fil=[]
fig,ax = plt.subplots(nrows=7, ncols=2, sharex=True, sharey=True)
xs=ax.flat
for n in range(len(nomes)):
  aux=[]
  auxErr=[]
  for i in range(len(dists)-1):
    aux.append(cy2[i][n])
    auxErr.append(cyErr2[i][n])
    item = idx_x[n]
    if (dfx.loc[item][' CC'] == 'NCC') :
      marca='o'
    if (dfx.loc[item][' CC'] == 'CC') :
      marca='d'

  fit_giDist2Fil.append(optimize.curve_fit(reta,cx,aux))
  x_fake=np.linspace(min(cx),max(cx),100)

  axis=xs[n]
#   axis.errorbar(cx, aux, xerr=cxErr, yerr=auxErr, marker=marca, capsize=2, mec='k', mfc=colors[n], \
# ms=6, elinewidth=1, ecolor=colors[n],linestyle='-',color=colors[n])
  axis.errorbar(cx, aux, xerr=cxErr, yerr=auxErr, marker=marca, capsize=2, mec='k', mfc=colors[n], \
ms=6, elinewidth=1, ecolor=colors[n],linestyle=' ',color=colors[n])
  axis.plot(x_fake,  reta(x_fake,fit_giDist2Fil[n][0][0],fit_giDist2Fil[n][0][1])  , linestyle='-',color=colors[n])

  tit=nomes[n]
  axis.set_title(tit, fontsize=8, fontweight='bold', pad=-10)
  axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
  axis.set_ylim(0.9,1.2)
  axis.set_xlim(0.45,1.55)
ylab=r'$(g-i)_{fil}$'
xlab='Distance to filament (Mpc)'
fig.text(0.43, 0.05, xlab, va='center')
fig.text(0.07, 0.5, ylab, va='center', rotation='vertical')
fig.set_size_inches(12.5, 8.5)
outname='Dist2Fil_giFil_fatiasIndividual.png'
plt.savefig(outname)
plt.close()

#INCLINAÇÃO DO AJUSTE DE GI_DIST2FIL X DENSIDADE RELATIVA
# cy=[]
# cyErr=[]
# for n in range(len(nomes)):
#   cy.append(fit_giDist2Fil[n][0][0])
#   cyErr.append(np.sqrt(fit_giDist2Fil[n][1][0][0]))
# cx=[]
# for n in range(len(nomes)):
#   cx.append(ProprFil[n][' Relative Density'].values[0])
# cxErr = [0]*14

# cx=f_agl[' redshift'].values
# for n in range(len(nomes)):
#   cx.append(ProprFil[n][' Relative Density'].values[0])
# cxErr = [0]*14

# ylab=r'Angular fit $(g-i)_{fil}$'
# xlab=r'$\rho$'
# plota_CC_HD(cx, cxErr,cy, cyErr, f_agl, xlab,ylab)

# x_fake=np.linspace(min(cx),max(cx),100)
# fit=optimize.curve_fit(reta,cx,cy,p0=[-0.1,-0.1])
# plt.plot(cx,reta(np.array(cx),0.01223,1.069))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# TEMPERATURA AGL X COR FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cx=f_agl[' kTm'].values
cxErr=((f_agl[' kTm+'].values - cx) + (cx - f_agl[' kTm-'].values ))/2

#temperatura media X G -R => CLEAN=================
cy=[]
cyErr=[]
cx2=[]
cy2=[]
for n in range(len(nomes)):
  cy.append(np.mean(GalsInFil_clean02[n]['gi']))
  cyErr.append(np.std(GalsInFil_clean02[n]['gi'])/np.sqrt(len(GalsInFil_clean[n]['gi'])))
  if f_agl['HighDens'][n]==1:
    cx2.append(cx[n])
    cy2.append(cy[n])

x_label=r'$kT_{agl}$ (keV)'
y_label=r'$(g-i)_{fil}$'

plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
# plota_status(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plt.title(r'$(g-i)_{fil} \times T_{agl}$', fontsize=10)

xfake=np.linspace(min(cx)-1,max(cx)+1, 100)
pars2=[]
p1=[]
p2=[]
std=[]
for _ in range(50):
  p=RobustLinear(cx2,cy2)
  pars2.append(p)
  p1.append(p[0])
  p2.append(p[1])
  std.append(p[3])

p1m=np.mean(p1)
p2m=np.mean(p2)
p1Err=np.sqrt((np.std(p1)/np.sqrt(len(p1)))**2 + (np.mean(std)/np.sqrt(len(std)))**2)
p2Err=np.sqrt(np.std(p2)/np.sqrt(len(p2))**2 + (np.mean(std)/np.sqrt(len(std)))**2)

yfit2=reta(xfake,p1m,p2m)
yfit2S=reta(xfake,p1m+p1Err,p2m+p2Err)
yfit2I=reta(xfake,p1m-p1Err,p2m-p2Err)
plt.plot(xfake,yfit2,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfit2I, yfit2S[::-1]),
    color='gray',alpha=0.5,edgecolor='w'
)
plt.xlim(4.5,8.5)
plt.ylim(0.95,1.2)

fit_lab=r'(%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (p1m,p1Err,p2m,p2Err)
plt.text(4.7, 0.97, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=1'))

outname='kT_giFil_clean.png'
plt.savefig(outname)
plt.close()

#temperatura media X G -R => closest to Cluster=================
cy=[]
cyErr=[]
dists=[1.,2.,3.,4.,5.,6.]
for i in dists:
  auxi=[]
  auxiEr=[]
  for n in range(len(nomes)):
    aux=GalsInFil_clean[n].loc[GalsInFil_clean[n].Dist2ClosCluster_Mpc <= i]
    auxi.append(np.mean(aux.gi))
    auxiEr.append(np.std(aux.gi)/np.sqrt(len(aux.gi)))
  cy.append(auxi)
  cyErr.append(auxiEr)

fig, ax=plt.subplots(nrows=2,ncols=3, sharex=True, sharey=True)
xs=ax.flat
for n in range(len(dists)):
  axis=xs[n]
  axis.errorbar(cx, cy[n], xerr=cxErr, yerr=cyErr[n], marker='o', capsize=2, mec='k', mfc=colors[n], \
ms=6, elinewidth=1, ecolor=colors[n],linestyle=' ')

  tit=r'Dist2Agl $<$' + str(dists[n]) + 'Mpc'
  axis.set_title(tit, fontsize=8, fontweight='bold')
  axis.tick_params(direction='in',labelsize=8)

ylab=r'$(g-i)_{fil}$'
xlab=r'kT$_{agl}$ (keV)'
fig.text(0.43, 0.05, xlab, va='center')
fig.text(0.07, 0.5, ylab, va='center', rotation='vertical')

outname='kT_giFil_Dist2Cl.png'
plt.savefig(outname)
plt.close()

#temperatura media X G -R => closest to Fil=================

cy=[]
cyErr=[]
dists=[0.5,1.,1.5,]
for i in dists:
  auxi=[]
  auxiEr=[]
  for n in range(len(nomes)):
    aux=GalsInFil_clean[n].loc[GalsInFil_clean[n].Dist2Fil_Mpc <= i]
    auxi.append(np.mean(aux.gi))
    auxiEr.append(np.std(aux.gi)/np.sqrt(len(aux.gi)))
  cy.append(auxi)
  cyErr.append(auxiEr)

fig, ax=plt.subplots(nrows=1,ncols=3, sharex=True, sharey=True)
xs=ax.flat
for n in range(len(dists)):
  axis=xs[n]
  axis.errorbar(cx, cy[n], xerr=cxErr, yerr=cyErr[n], marker='o', capsize=2, mec='k', mfc=colors[n], \
ms=6, elinewidth=1, ecolor=colors[n],linestyle=' ')

  tit=r'Dist2Fil $<$' + str(dists[n]) + 'Mpc'
  axis.set_title(tit, fontsize=8, fontweight='bold')
  axis.tick_params(direction='in',labelsize=8)

ylab=r'$(g-i)_{fil}$'
xlab=r'kT$_{agl}$ (keV)'
fig.text(0.43, 0.05, xlab, va='center')
fig.text(0.07, 0.5, ylab, va='center', rotation='vertical')
fig.set_size_inches(12.5,4.5)
outname='kT_giFil_Dist2Fil.png'
plt.savefig(outname)
plt.close()

#temperatura media X G -R => closest to Cluster | FATIA =================
nomcy='Dist2ClosCluster_Mpc'
x_label=r'kT$_{agl}$ (keV)'
y_label=r'$(g-i)_{fil}$'
filey=GalsInFil_clean
plota_dist2cluster_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey)

outname='kT_giFil_Dist2Cl_fatia.png'
plt.savefig(outname)
plt.close()

#temperatura media X G -R => closest to Fil | FATIA =================
nomcy='Dist2Fil_Mpc'
x_label=r'kT$_{agl}$ (keV)'
y_label=r'$(g-i)_{fil}$'
filey=GalsInFil_clean
plota_dist2fil_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey)

outname='kT_giFil_Dist2Fil_fatia.png'
plt.savefig(outname)
plt.close()

#temperatura media X G -R => closest to Cl FATIA =================
cy=[]
cyErr=[]
for n in range(len(nomes)):
  auxi=[]
  auxiEr=[]
  aux1=GalsInFil_clean[n].query('1.5<Dist2ClosCluster_Mpc<= 3')
  auxi.append(np.mean(aux1.gi))
  auxiEr.append(np.std(aux1.gi)/np.sqrt(len(aux1.gi)))  
  cy.append(auxi)
  cyErr.append(auxiEr)

x_label=r'$kT_{agl}$ (keV)'
y_label=r'$(g-i)_{fil}$'

# plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plota_status(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plt.title(r'$(g-i)_{fil} \times T_{agl}$, Dist2Agl <= 3', fontsize=10)
outname='kT_giFil_innerCl3.png'
plt.savefig(outname)
plt.close()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# REDSHIFT X COR FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cx=f_agl[' redshift'].values
cxErr=[0]*len(cx)

# REDSHIFT X COR TOTAL FIL --------------------------------------
cx2=[]
cy2=[]
cy2Err=[]
cy=[]
cyErr=[]
cxErr=[]
cx2Err=[]
for n in range(len(nomes)):
  cy.append(np.mean(GalsInFil_clean02[n]['gi']))
  cyErr.append(np.std(GalsInFil_clean02[n]['gi'])/np.sqrt(len(GalsInFil_clean02[n]['gi'])))
  cxErr.append((0.03*(1+cx[n]))/2)
  # cy.append(np.mean(GalsInFil_clean[n]['gi']))
  # cyErr.append(np.std(GalsInFil_clean[n]['gi'])/np.sqrt(len(GalsInFil_clean[n]['gi'])))
  if f_agl['HighDens'][n]==1:
    cx2.append(cx[n])
    cx2Err.append(cxErr[n])
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])



x_label=r'$z_{agl}$'
y_label=r'$(g-i)_{fil}$'

# plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)


xfake=np.linspace(min(cx)-0.1, max(cx)+0.1,100)
#reta - tudo
# ssy=np.array(cy2Err).ravel()
# ssx=np.array(cx2Err).ravel()
# cx2=np.array(cx2).ravel()
# cy2=np.array(cy2).ravel()
# # pars,errs=optimize.curve_fit(reta,cx,cy)
# pars,errs=fitReta(cx2,cy2,ssx,ssy)
# yfit=reta(xfake,pars[0],pars[1])
# yfitS=reta(xfake,pars[0]+np.sqrt(errs[0][0]),pars[1]+np.sqrt(errs[1][1]))
# yfitI=reta(xfake,pars[0]-np.sqrt(errs[0][0]),pars[1]-np.sqrt(errs[1][1]))
# plt.plot(xfake,yfit,color='grey')
# # plt.fill(
# #     np.append(xfake, xfake[::-1]),
# #     np.append(yfitI, yfitS[::-1]),
# #     color='gray',alpha=0.5,edgecolor='w'
# # )
# plt.plot(xfake,yfit, color='gray', label = 'Fit total')
# fit_lab=r'Total: (%.2f$\pm$ %.2f)X + (%.2f$\pm$ %.2f)' % (pars[0],np.sqrt(errs[0][0]),pars[1],np.sqrt(errs[1][1]))
# plt.text(0.15, 1.22, fit_lab, va='center',color='grey',\
#   bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=1', alpha=0.5))
# plt.text(0.15, 1.22, fit_lab, va='center',color='grey',\
#   bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=1', alpha=0.5))

#reta - clean
ssy=np.array(cy2Err).ravel()
ssx=np.array(cx2Err).ravel()
cx2=np.array(cx2).ravel()
cy2=np.array(cy2).ravel()
pars,errs=fitReta(cx2,cy2,ssx,ssy)
yfit=reta(xfake,pars[0],pars[1])
yfitS=reta(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=reta(xfake,pars[0]-errs[0],pars[1]-errs[1])
plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='k',alpha=0.2,edgecolor='w'
)
fit_lab=r'Clean: (%.2f$\pm$ %.2f)X + (%.2f$\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(0.15, 1.22, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='k', boxstyle='round,pad=1', alpha=0.5))


plt.ylim(0.95,1.25)
plt.xlim(0.135,0.355)
plt.title(r'$(g-i)_{fil} \times z$', fontsize=10)
outname='z_giFil_tot.png'
plt.savefig(outname)
plt.close()


## REDSHIFT X COR FIL => closest to Cluster | FATIA =================
nomcy='Dist2ClosCluster_Mpc'
x_label=r'$z_{agl}$'
y_label=r'$(g-i)_{fil}$'
filey=GalsInFil_clean02
plota_dist2cluster_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey)
plt.xlim(0.135,0.355)
outname='z_giFil_Dist2Cl_fatia.png'
plt.savefig(outname)
plt.close()

## REDSHIFT X COR FIL => closest to Fil | FATIA =================
nomcy='Dist2Fil_Mpc'
x_label=r'$z_{agl}$'
y_label=r'$(g-i)_{fil}$'
filey=GalsInFil_clean02
plota_dist2fil_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey)
plt.xlim(0.135,0.355)
outname='z_giFil_Dist2Fil_fatia.png'
plt.savefig(outname)
plt.close()


## REDSHIFT X COR FIL/ COR AGL => INNER =================
cy=[]
cyErr=[]
for n in range(len(nomes)):
  # aux1=GalsInFil_clean[n].loc[(GalsInFil_clean[n].Dist2ClosCluster_Mpc<= 2.5)]
  # aux2= aux1.loc[aux1.Dist2Fil_Mpc<= 1.]
  auxi=np.mean(GalsInFil_clean02[n].gi)
  auxiEr=np.std(GalsInFil_clean02[n].gi)/np.sqrt(len(GalsInFil_clean02[n].gi))
  auxc=np.mean(GalsInCl[n]['gi'])
  auxcErr=np.std(GalsInCl[n]['gi'])/np.sqrt(len(GalsInCl[n]['gi']))
  cy.append(auxi/auxc)
  cyErr.append( np.sqrt( (auxiEr/auxc)**2 + (auxi*auxcErr/(auxc**2))**2 ) )

x_label=r'$z_{agl}$'
y_label=r'$(g-i)_{fil} / (g-i)_{agl}$'

# plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plt.xlim(0.135,0.355)
plt.hlines(y=1,xmin=0.135,xmax=0.355, linestyle='--', color='gray')
# plt.title(r'$(g-i) \frac{fil}{agl} \times z$', fontsize=10)
outname='z_giFilCl.png'
plt.savefig(outname)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cx=f_agl[' redshift'].values
cxErr=[0]*len(cx)

# REDSHIFT X COR TOTAL FIL --------------------------------------
cy=[]
cyErr=[]
cy2=[]
cy2Err=[]
cx2=[]
cx2Err=[]
for n in range(len(nomes)):
  cy.append(np.mean(GalsInFil_clean[n]['gi']))
  cyErr.append(np.std(GalsInFil_clean[n]['gi'])/np.sqrt(len(GalsInFil_clean[n]['gi'])))
  if f_agl['HighDens'][n] ==1:
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
    cx2.append(cx[n])
    cx2Err.append(cxErr[n])

x_label=r'$z_{agl}$'
y_label=r'$(g-i)_{fil}$'

# plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plt.ylim(0.85,1.4)

xfake=np.linspace(min(cx2)-0.1, max(cx2)+0.1,100)
#reta - tudo
ss=np.array(cy2Err).ravel()
cxx=np.array(cx2).ravel()
cyy=np.array(cy2).ravel()
pars,errs=optimize.curve_fit(reta,cxx,cyy)
yfit=reta(xfake,pars[0],pars[1])
yfitS=reta(xfake,pars[0]+np.sqrt(errs[0][0]),pars[1]+np.sqrt(errs[1][1]))
yfitI=reta(xfake,pars[0]-np.sqrt(errs[0][0]),pars[1]-np.sqrt(errs[1][1]))
plt.plot(xfake,yfit,color='grey')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='gray',alpha=0.5,edgecolor='w'
)
plt.plot(xfake,yfit, color='gray', label = 'Fit total')
fit_lab=r'Total: (%.2f$\pm$ %.2f)X + (%.2f$\pm$ %.2f)' % (pars[0],np.sqrt(errs[0][0]),pars[1],np.sqrt(errs[1][1]))
plt.text(0.15, 1.35, fit_lab, va='center',\
  bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=1', alpha=0.5))
plt.xlim(0.135,0.355)
plt.title(r'$(g-i)_{fil} \times z$', fontsize=10)
outname='z_giFil_HighDens.png'
plt.savefig(outname)
plt.close()



## REDSHIFT X COR FIL/ COR AGL => INNER =================
cy=[]
cyErr=[]
cy2=[]
cy2Err=[]
cx2=[]
cx2Err=[]
for n in range(len(nomes)):
  # aux1=GalsInFil_clean[n].loc[(GalsInFil_clean[n].Dist2ClosCluster_Mpc<= 2.5)]
  # aux2= aux1.loc[aux1.Dist2Fil_Mpc<= 1.]
  auxi=np.mean(GalsInFil_clean[n].gi)
  auxiEr=np.std(GalsInFil_clean[n].gi)/np.sqrt(len(GalsInFil_clean[n].gi))
  auxc=np.mean(GalsInCl[n]['gi'])
  auxcErr=np.std(GalsInCl[n]['gi'])/np.sqrt(len(GalsInCl[n]['gi']))
  cy.append(auxi/auxc)
  cyErr.append( np.sqrt( (auxiEr/auxc)**2 + (auxi*auxcErr/(auxc**2))**2 ) )
  if f_agl['HighDens'][n] ==1:
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
    cx2.append(cx[n])
    cx2Err.append(cxErr[n])

x_label=r'$z_{agl}$ (keV)'
y_label=r'$(g-i)_{fil} / (g-i)_{agl}$'

# plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plt.xlim(0.135,0.355)
plt.hlines(y=1,xmin=0.135,xmax=0.355, linestyle='--', color='gray')
plt.title(r'$(g-i) \frac{fil}{agl} \times z$', fontsize=10)
outname='z_giFilCl_HighDens.png'
plt.savefig(outname)
plt.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MASSA AGL X COR FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cx=f_agl[' Mtot_isoT(1e13Msun)'].values/10
cxErr=f_agl[' MtotErr'].values/10

## Mtot X COR FIL => closest to Cluster | FATIA =================
nomcy='Dist2ClosCluster_Mpc'
x_label=r'M$_{500}$ ($10^{14} M_{\odot}$)'
y_label=r'$(g-i)_{fil}$'
filey=GalsInFil_clean02
plota_dist2cluster_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey)
outname='Mtot_giFil_Dist2Cl_fatia.png'
plt.savefig(outname)
plt.close()

## Mtot e kT X COR FIL  =================

y_label=r'$(g-i)_{fil}$'

xM_label=r'M$_{500}$ ($10^{14} M_{\odot}$)'
cxM=f_agl[' Mtot_isoT(1e13Msun)'].values/10
cxMErr=f_agl[' MtotErr'].values/10

cxT=f_agl[' kTm'].values
cxTErr=((f_agl[' kTm+'].values - cx) + (cx - f_agl[' kTm-'].values ))/2
xT_label=r'kT$_{agl} (keV)$'

cy=[]
cyErr=[]
cy2=[]
cy2Err=[]
cxM2=[]
cxM2Err=[]
cxT2=[]
cxT2Err=[]
for n in range(len(nomes)):
  auxi=[]
  auxiEr=[]
  aux1=GalsInFil_clean02[n]
  auxi.append(np.mean(aux1.gi))
  auxiEr.append(np.std(aux1.gi)/np.sqrt(len(aux1.gi)))  
  cy.append(auxi)
  cyErr.append(auxiEr)
  if f_agl['HighDens'][n] ==1:
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
    cxM2.append(cxM[n])
    cxM2Err.append(cxMErr[n])
    cxT2.append(cxT[n])
    cxT2Err.append(cxTErr[n])

x_label=r'M$_{500}$ ($10^{14} M_{\odot}$)'
y_label=r'$(g-i)_{fil}$'

# plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
pp=[]
ee=[]
for i in range(2):
  if i ==0:
    xx = cxM2
    xxErr = cxM2Err
  else:
    xx = cxT2
    xxErr = cxT2Err

  ssy=np.array(cy2Err).ravel()
  ssx=np.array(xxErr).ravel()
  cx2=np.array(xx).ravel()
  cy2=np.array(cy2).ravel()
  pars,errs=fitReta(xx,cy2,ssx,ssy)
  pp.append(pars)
  ee.append(errs)

legs=[]
fig,ax=plt.subplots(nrows=1,ncols=2, sharey=True)
for i in range(2):
  axis=ax.flat[i]
  if i ==0:
    cx = cxM
    cxErr = cxMErr
    xlab=xM_label
    xp=1.95
    yp= 1.22
    yl=[0.95,1.25]
    xl=[1.5,8]
  else:
    cx = cxT
    cxErr = cxTErr
    xlab=xT_label
    xp=4.8
    yp= 1.22
    yl=[0.95,1.25]
    xl=[4.5,8.5]

  for n in range(len(nomes)):
    # yy=frac_red[n][coly]/frac_color[' RF_tot_CMP'][n]
    item=nomes[n]
    if (f_agl[' CC'][n] == 'NCC') :
      marca='o'
    elif (f_agl[' CC'][n] == 'CC') :
      marca='d'
    else :
        marca=''
    if f_agl['HighDens'][n] == 0:
      alp = 0.4
      meww=3
    else:
      alp=1
      meww=1

    axis.errorbar(cx[n], cy[n], xerr=cxErr[n], yerr=cyErr[n], marker=marca, capsize=2, mec='k', mfc=colors[n], \
  ms=6, elinewidth=1, ecolor=colors[n],alpha=alp,mew=meww)

    axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)

    if i==0:
      legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[n], mec='k', markersize=9, alpha=0.7, label=item))

  xfake=np.linspace(min(cx)-10, max(cx)+10,100)
  yfit=reta2(pp[i],xfake)
  yfitS=reta(xfake,pp[i][0]+ee[i][0],pp[i][1]+ee[i][1])
  yfitI=reta(xfake,pp[i][0]-ee[i][0],pp[i][1]-ee[i][1])
  axis.plot(xfake,yfit,color='gray')
  axis.fill(
      np.append(xfake, xfake[::-1]),
      np.append(yfitI, yfitS[::-1]),
      color='dimgray',alpha=0.3,edgecolor='w'
  )

  axis.set_xlabel(xlab)
  axis.set_xlim(xl)
  axis.set_ylim(yl)

  fit_lab=r'(%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pp[i][0],ee[i][0],pp[i][1],ee[i][1])
  axis.text(xp,yp, fit_lab, va='center',color='k', fontsize=9,\
    bbox=dict(facecolor='none', edgecolor='k', boxstyle='round,pad=1'))


legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='NCC'))
legs.append(Line2D([0], [0], marker='d', color='w',markerfacecolor="k", markersize=7, alpha=0.7, label='CC'))

fig.set_size_inches(8.5,4.5)
fig.subplots_adjust(wspace=0.15, top=0.87)
fig.legend(handles=legs, loc='upper center', bbox_to_anchor=(0.51,1.),
          ncol=6, fancybox=True, shadow=True, fontsize = 'x-small')


fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')


outname='Mtot_kT_giFil.png'
plt.savefig(outname)
plt.close()




## Mtot X COR FIL => closest to Fil | FATIA =================
nomcy='Dist2Fil_Mpc'
x_label=r'M$_{agl}^{T}$ ($10^{14} M_{\odot}$)'
y_label=r'$(g-i)_{fil}$'
filey=GalsInFil_clean02
plota_dist2fil_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey)
outname='Mtot_giFil_Dist2Fil_fatia.png'
plt.savefig(outname)
plt.close()

## Mtot X COR FIL = =================
cy=[]
cyErr=[]
cy2=[]
cy2Err=[]
cx2=[]
cx2Err=[]
for n in range(len(nomes)):
  auxi=[]
  auxiEr=[]
  aux1=GalsInFil_clean02[n]
  auxi.append(np.mean(aux1.gi))
  auxiEr.append(np.std(aux1.gi)/np.sqrt(len(aux1.gi)))  
  cy.append(auxi)
  cyErr.append(auxiEr)
  if f_agl['HighDens'][n] ==1:
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
    cx2.append(cx[n])
    cx2Err.append(cxErr[n])

x_label=r'M$_{agl}^{T}$ ($10^{14} M_{\odot}$)'
y_label=r'$(g-i)_{fil}$'

# plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)

ssy=np.array(cy2Err).ravel()
ssx=np.array(cx2Err).ravel()
cx2=np.array(cx2).ravel()
cy2=np.array(cy2).ravel()
xfake=np.linspace(min(cx)-1, max(cx)+1,100)
pars,errs=fitReta(cx2,cy2,ssx,ssy)

yfit=reta2(pars,xfake)
yfitS=reta(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=reta(xfake,pars[0]-errs[0],pars[1]-errs[1])
plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='k',alpha=0.2,edgecolor='w'
)

fit_lab=r'Clean: (%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(1.85, 1.22, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='k', boxstyle='round,pad=1'))
plt.ylim(0.95,1.25)
plt.xlim(1.5,8)


outname='Mtot_giFil_tot.png'
plt.savefig(outname)
plt.close()


## Mtot X COR FIL => INNER =================
cy=[]
cyErr=[]
cy2=[]
cy2Err=[]
cx2=[]
cx2Err=[]
for n in range(len(nomes)):
  auxi=[]
  auxiEr=[]
  aux1=GalsInFil_clean02[n].query('Dist2ClosCluster_Mpc <=3')
  auxi.append(np.mean(aux1.gi))
  auxiEr.append(np.std(aux1.gi)/np.sqrt(len(aux1.gi)))  
  cy.append(auxi)
  cyErr.append(auxiEr)
  if f_agl['HighDens'][n] ==1:
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
    cx2.append(cx[n])
    cx2Err.append(cxErr[n])

x_label=r'M$_{agl}^{T}$ ($10^{14} M_{\odot}$)'
y_label=r'$(g-i)_{fil}$'

# plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)

ssy=np.array(cy2Err).ravel()
ssx=np.array(cx2Err).ravel()
cx2=np.array(cx2).ravel()
cy2=np.array(cy2).ravel()
xfake=np.linspace(min(cx)-1, max(cx)+1,100)
pars,errs=fitReta(cx2,cy2,ssx,ssy)

yfit=reta2(pars,xfake)
yfitS=reta(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=reta(xfake,pars[0]-errs[0],pars[1]-errs[1])
plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='k',alpha=0.2,edgecolor='w'
)

fit_lab=r'Clean: (%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(1.85, 1.22, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='k', boxstyle='round,pad=1'))
plt.ylim(0.95,1.25)
plt.xlim(1.5,8)

outname='Mtot_giFil_innerCl3.png'
plt.savefig(outname)
plt.close()

## Mtot  X COR FIL/ COR AGL  =================
cy=[]
cyErr=[]

cy2=[]
cy2Err=[]
cx2=[]
cx2Err=[]

cyCC=[]
cyCCErr=[]
cxCC=[]
cxCCErr=[]
for n in range(len(nomes)):
  auxi=np.mean(GalsInFil_clean02[n].gi)
  auxiEr=np.std(GalsInFil_clean02[n].gi)/np.sqrt(len(GalsInFil_clean02[n].gi))
  auxc=np.mean(GalsInCl[n]['gi'])
  auxcErr=np.std(GalsInCl[n]['gi'])/np.sqrt(len(GalsInCl[n]['gi']))
  cy.append(auxi/auxc)
  cyErr.append( np.sqrt( (auxiEr/auxc)**2 + (auxi*auxcErr/(auxc**2))**2 ) )
  if f_agl['HighDens'][n] ==1:
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
    cx2.append(cx[n])
    cx2Err.append(cxErr[n])
    if f_agl[' CC'][n] == 'NCC':
      cyCC.append(cy[n])
      cyCCErr.append(cyErr[n])
      cxCC.append(cx[n])
      cxCCErr.append(cxErr[n])


x_label=r'M$_{500}$ ($10^{14} M_{\odot}$)'
y_label=r'$(g-i)_{fil} / (g-i)_{agl}$'

plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)

ssy=np.array(cy2Err).ravel()
ssx=np.array(cx2Err).ravel()
cx2=np.array(cx2).ravel()
cy2=np.array(cy2).ravel()
xfake=np.linspace(min(cx)-1, max(cx)+1,100)
pars,errs=fitReta(cx2,cy2,ssx,ssy)

yfit=reta2(pars,xfake)
yfitS=reta(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=reta(xfake,pars[0]-errs[0],pars[1]-errs[1])
plt.plot(xfake,yfit,color='gray', linestyle='-', lw=1)
# plt.fill(
#     np.append(xfake, xfake[::-1]),
#     np.append(yfitI, yfitS[::-1]),
#     color='k',alpha=0.2,edgecolor='w'
# )

fit_lab=r'(%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(1.9, 1.035, fit_lab, va='center',color='gray',fontsize=9,\
  bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=1'))
plt.text(1.9, 1.035, fit_lab, va='center',color='gray',fontsize=9,\
  bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=1'))

ssy=np.array(cyCCErr).ravel()
ssx=np.array(cxCCErr).ravel()
cxCC=np.array(cxCC).ravel()
cyCC=np.array(cyCC).ravel()
xfake=np.linspace(min(cx)-1, max(cx)+1,100)
pars,errs=fitReta(cxCC,cyCC,ssx,ssy)

yfit=reta2(pars,xfake)
yfitS=reta(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=reta(xfake,pars[0]-errs[0],pars[1]-errs[1])
plt.plot(xfake,yfit,color='k',lw=1)
# plt.fill(
#     np.append(xfake, xfake[::-1]),
#     np.append(yfitI, yfitS[::-1]),
#     color='k',alpha=0.2,edgecolor='w'
# )

fit_lab=r'(%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(1.9, 0.865, fit_lab, va='center',color='k',fontsize=9,\
  bbox=dict(facecolor='none', edgecolor='k', boxstyle='round,pad=1'))


# plota_status(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
# plt.hlines(y=1,xmin=min(cx)-2,xmax=max(cx)+2, linestyle='--', color='gray')
plt.xlim(1.5,8.2)
plt.ylim(0.85,1.05)
# plt.title(r'$(g-i) \frac{fil}{agl} \times M_{agl}^T$', fontsize=10)
outname='Mtot_giFilCl.png'
plt.savefig(outname)
plt.close()

## Mtot  X COR FIL/ COR AGL INNER =================
cy=[]
cyErr=[]

cy2=[]
cy2Err=[]
cx2=[]
cx2Err=[]

cyCC=[]
cyCCErr=[]
cxCC=[]
cxCCErr=[]
for n in range(len(nomes)):
  dd=GalsInFil_clean02[n].query('Dist2ClosCluster_Mpc <=3')
  auxi=np.mean(dd.gi)
  auxiEr=np.std(dd.gi)/np.sqrt(len(dd.gi))
  auxc=np.mean(GalsInCl[n]['gi'])
  auxcErr=np.std(GalsInCl[n]['gi'])/np.sqrt(len(GalsInCl[n]['gi']))
  cy.append(auxi/auxc)
  cyErr.append( np.sqrt( (auxiEr/auxc)**2 + (auxi*auxcErr/(auxc**2))**2 ) )
  if f_agl['HighDens'][n] ==1:
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
    cx2.append(cx[n])
    cx2Err.append(cxErr[n])
    if f_agl[' CC'][n] == 'NCC':
      cyCC.append(cy[n])
      cyCCErr.append(cyErr[n])
      cxCC.append(cx[n])
      cxCCErr.append(cxErr[n])


x_label=r'M$_{agl}^{T}$ ($10^{14} M_{\odot}$)'
y_label=r'$(g-i)_{fil} / (g-i)_{agl}$'

plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)

ssy=np.array(cy2Err).ravel()
ssx=np.array(cx2Err).ravel()
cx2=np.array(cx2).ravel()
cy2=np.array(cy2).ravel()
xfake=np.linspace(min(cx)-1, max(cx)+1,100)
pars,errs=fitReta(cx2,cy2,ssx,ssy)

yfit=reta2(pars,xfake)
yfitS=reta(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=reta(xfake,pars[0]-errs[0],pars[1]-errs[1])
plt.plot(xfake,yfit,color='k', linestyle=':')
# plt.fill(
#     np.append(xfake, xfake[::-1]),
#     np.append(yfitI, yfitS[::-1]),
#     color='k',alpha=0.2,edgecolor='w'
# )

fit_lab=r'Clean: (%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(1.85, 1.03, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='k', boxstyle='round,pad=1'))

ssy=np.array(cyCCErr).ravel()
ssx=np.array(cxCCErr).ravel()
cxCC=np.array(cxCC).ravel()
cyCC=np.array(cyCC).ravel()
xfake=np.linspace(min(cx)-1, max(cx)+1,100)
pars,errs=fitReta(cxCC,cyCC,ssx,ssy)

yfit=reta2(pars,xfake)
yfitS=reta(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=reta(xfake,pars[0]-errs[0],pars[1]-errs[1])
plt.plot(xfake,yfit,color=colors[0])
# plt.fill(
#     np.append(xfake, xfake[::-1]),
#     np.append(yfitI, yfitS[::-1]),
#     color='k',alpha=0.2,edgecolor='w'
# )

fit_lab=r'NCC: (%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(1.85, 0.87, fit_lab, va='center',color=colors[0],\
  bbox=dict(facecolor='none', edgecolor=colors[0], boxstyle='round,pad=1'))


# plota_status(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plt.hlines(y=1,xmin=min(cx)-2,xmax=max(cx)+2, linestyle='--', color='gray')
plt.xlim(1.5,8.5)
plt.ylim(0.85,1.05)
# plt.title(r'$(g-i) \frac{fil}{agl} \times M_{agl}^T$', fontsize=10)
outname='Mtot_giFilCl.png'
plt.savefig(outname)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# COR AGL X COR FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cx=[]
cxErr=[]
for n in range(len(nomes)):
  cx.append(np.mean(GalsInCl[n]['gi']))
  cxErr.append(np.std(GalsInCl[n]['gi'])/np.sqrt(len(GalsInCl[n]['gi'])))

## COR AGL X COR FIL => closest to Cluster | FATIA =================
nomcy='Dist2ClosCluster_Mpc'
x_label=r'$(g-i)_{agl}$'
y_label=r'$(g-i)_{fil}$'
filey=GalsInFil_clean
plota_dist2cluster_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey)
outname='giCl_giFil_Dist2Cl_fatia.png'
plt.savefig(outname)
plt.close()

## COR AGL X COR FIL => closest to Fil | FATIA =================
nomcy='Dist2Fil_Mpc'
x_label=r'$(g-i)_{agl}$'
y_label=r'$(g-i)_{fil}$'
# filey=GalsInFil_clean
# plota_dist2fil_fatia(cx,cxErr,nomcy,x_label, y_label, nomes, filey)

cx2=[]
cx2Err=[]
for n in range(len(nomes)):
  if f_agl['HighDens'][n]==1:
    cx2.append(cx[n])
    cx2Err.append(cxErr[n])

cy=[]
cyErr=[]
dists=[0,0.5,1.,1.5]
for i in range(len(dists)-1):
  auxi=[]
  auxiEr=[]
  for n in range(len(nomes)):
    if f_agl['HighDens'][n]==1:
      aux1=GalsInFil_clean02[n].loc[(dists[i] < GalsInFil_clean02[n]['Dist2Fil_Mpc'])]
      aux2= aux1.loc[aux1[nomcy]<= dists[i+1]]
      auxi.append(np.mean(aux2.gi))
      auxiEr.append(np.std(aux2.gi)/np.sqrt(len(aux2.gi)))
  cy.append(auxi)
  cyErr.append(auxiEr)

fig, ax=plt.subplots(nrows=1,ncols=3, sharex=True, sharey=True)
xs=ax.flat
for n in range(len(dists)-1):
  axis=xs[n]
  axis.errorbar(cx2, cy[n], xerr=cx2Err, yerr=cyErr[n], marker='o', capsize=2, mec='k', mfc=colors[n], \
ms=6, elinewidth=1, ecolor=colors[n],linestyle=' ')

  tit=str(dists[n]) + r'$<$ Dist2Fil $<=$' + str(dists[n+1]) + 'Mpc'
  axis.set_title(tit, fontsize=8, fontweight='bold')
  axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)

fig.text(0.5, 0.05, x_label, va='center')
fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')
fig.set_size_inches(12.5,4.5)
plt.show()

outname='giCl_giFil_Dist2Fil_fatia_HighDens.png'
plt.savefig(outname)
plt.close()

## COR AGL X COR FIL => INNER =================
cy=[]
cyErr=[]
for n in range(len(nomes)):
  auxi=[]
  auxiEr=[]
  aux1=GalsInFil_clean[n].query('1.5<Dist2ClosCluster_Mpc<= 3')
  auxi.append(np.mean(aux1.gi))
  auxiEr.append(np.std(aux1.gi)/np.sqrt(len(aux1.gi)))  
  cy.append(auxi)
  cyErr.append(auxiEr)

x_label=r'$(g-i)_{agl}$'
y_label=r'$(g-i)_{fil}$'

plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
#plota_status(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plt.title(r'$(g-i)_{fil} \times (g-i)_{agl}$, Dist2Agl <= 3', fontsize=10)
outname='giFilCl_innerCl3.png'
plt.savefig(outname)
plt.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PROPS FIL X COR FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cy=[]
cyErr=[]
for n in range(len(nomes)):
  cy.append(np.mean(GalsInFil_clean02[n]['gi']))
  cyErr.append(np.std(GalsInFil_clean02[n]['gi'])/np.sqrt(len(GalsInFil_clean02[n]['gi'])))

cy2=[]
cxL=[]
cxLErrS=[]
cxLErrI=[]
cxP=[]
cxPErrS=[]
cxPErrI=[]
cxL2=[]
cxP2=[]
cxP2Err=[]
cxL2Err=[]
for n in range(len(nomes)):
  cxL.append(ProprFil[n][' Length (Mpc)'])
  cxP.append(ProprFil[n][' Relative Density'])
  # cxLErr.append( (  ProprFil[n][' LengthErrInf']+ ProprFil[n][' LengthErrInf'] )/2 )
  cxLErrS.append( ProprFil[n][' LengthErrSup'])
  cxLErrI.append( ProprFil[n][' LengthErrInf'])
  # cxPErr.append( (  ProprFil[n][' Relative DensityErrInf']+ ProprFil[n][' Relative DensityErrSup'] )/2 )
  cxPErrS.append( ProprFil[n][' Relative DensityErrSup'])
  cxPErrI.append( ProprFil[n][' Relative DensityErrInf'])

  if f_agl['HighDens'][n] ==1:
    cxL2.append(cxL[n])
    cxP2.append(cxP[n])
    cxL2Err.append(cxLErrS[n])
    cxP2Err.append(cxPErrS[n])
    cy2.append(cy[n])

cxLErr = [0]*14
cxPErr = [0]*14


#COMPRRIMENTO X COR TOTAL-----------------------------------
x_label=r'$L_{fil}$ (Mpc)'
y_label=r'$(g-i)_{fil}$'
plota_CC_HD(cxL, cy, cxLErr, cyErr, f_agl, x_label, y_label)
outname='Len_giFil_HighDens.png'
plt.savefig(outname)
plt.close()

# DENSI REL X COR TOTAL-----------------------------------
x_label= r'$\rho_{rel}$'
y_label=r'$(g-i)_{fil}$'
plota_CC(cxP, cy, [0]*14, cyErr, f_agl, x_label, y_label)


sc=stats.sigmaclip(cy,2.5,2.45)

gv=sc[0]
low=sc[1]
high=sc[2]
avrg=np.mean(gv)
std=np.std(gv)

xmin=min(np.array(cxP))-2
xmax=max(np.array(cxP)) + 2
plt.hlines(high,xmin,xmax, linestyle=':', color='gray')
plt.hlines(low,xmin,xmax, linestyle=':', color='gray', label=r'$2\sigma limits$')
plt.hlines(avrg,xmin,xmax, linestyle='-', color='gray',label=r'$GV = 0.89 \pm 0.01$')
plt.xlim(xmin,xmax)
plt.axhspan(low,high, color='dimgray', alpha=0.35, zorder=0)

# legs=[]
# legs.append(Line2D([0], [0], color='green', lw=1, label=r'$\overline{(g-i)} = %.2f \pm %.2f$' %(avrg,std)))
# legs.append(Line2D([0], [0], color='r', lw=1, label=r'$2.5\sigma-clipping$'))

# plt.subplots_adjust(hspace=0, right=0.7)
# plt.legend(handles=legs, loc='best', fancybox=True, shadow=True, fontsize = 'x-small')


# plt.title(r'$(g-i)_{fil} \times $ Relative density', fontsize=10)
outname='DensiRel_giFil_SigmaClip.png'
plt.savefig(outname)
plt.close()

#-------------------------------------------------
# cy=[]
# cyErr=[]
# for n in range(len(nomes)):
#   auxi=[]
#   auxiEr=[]
#   aux1=GalsInFil_clean[n].query('1.5<Dist2ClosCluster_Mpc<= 3.')
#   auxi.append(np.mean(aux1.gi))
#   auxiEr.append(np.std(aux1.gi)/np.sqrt(len(aux1.gi)))  
#   cy.append(auxi)
#   cyErr.append(auxiEr)

# #COMPRRIMENTO X COR INNER-----------------------------------
# x_label='Filament Length (Mpc)'
# y_label=r'$(g-i)_{fil}$'
# plota_CC_HD(cxL, cy, cxLErr, cyErr, f_agl, x_label, y_label)
# plt.title(r'$(g-i)_{fil} \times$ Filament Lenght, Dist2Agl <= 3', fontsize=10)
# outname='Len_giFil_innerCl3_HighDens.png'
# plt.savefig(outname)
# plt.close()

# # DENSI REL X COR INNER-----------------------------------
# x_label= r'$\rho_{fil}/\rho_{field}$'
# y_label=r'$(g-i)_{fil}$'
# plota_CC_HD(cxP, cy, cxPErr, cyErr, f_agl, x_label, y_label)
# plt.title(r'$(g-i)_{fil} \times$ Relative density', fontsize=10)
# outname='DensiRel_giFil_innerCl3_HighDens.png'
# plt.savefig(outname)
# plt.close()


# DENSI REL X COMPRI-----------------------------------
def f(x,a,b):
  func=a/(x) + b
  return func

x_label= r'$\rho_{rel}$'
y_label=r'$L_{fil}$ (Mpc)'

# plota_CC_HD(cxP, cxL, [0]*14, [0]*14, f_agl, x_label, y_label)
plota_CC(cxP, cxL,[0]*14, [0]*14, f_agl, x_label, y_label)


# cxP2=np.array(cxP2).ravel()
# cxL2=np.array(cxL2).ravel()
cxP=np.array(cxP).ravel()
cxL=np.array(cxL).ravel()
cxLErr=(( np.array(cxLErrS)+np.array(cxLErrI) ) /2).ravel()
cxPErr=(( np.array(cxPErrS)+np.array(cxPErrI) ) /2).ravel()
cxLErrI=np.array(cxLErrI).ravel()
cxLErrS=np.array(cxLErrS).ravel()

pars,errs=fitcurva(cxP,cxL,cxPErr,cxLErr)

xfake=np.linspace(min(cxP)-2,max(cxP)+2,100)
yfit=f(xfake,pars[0],pars[1])
yfitS=f(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=f(xfake,pars[0]-errs[0],pars[1]-errs[1])

plt.plot(xfake,yfit,color='gray')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='dimgray',alpha=0.35,edgecolor='w'
)

plt.xlim(2.,20)
plt.ylim(0,200)

fit_lab=r' $\frac{(%i \pm %i)}{X}$ + (%i $\pm$ %i)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(11,180, fit_lab, va='center',\
  bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=1', alpha=0.5))


# plt.ylim(0,180)
# plt.xlim(10,80)
# plt.title(r'Filament Lenght $\times$ Relative density ', fontsize=10)
outname='DensiRel_Len_HighDens.png'
plt.savefig(outname)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DIST X PROPS FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cyL=[]
cyLErr=[]
cyP=[]
cyPErr=[]
for n in range(len(nomes)):
  cyL.append(ProprFil[n][' Length (Mpc)'])
  cyP.append(ProprFil[n][' Relative Density'])
cyLErr = [0]*14
cyPErr = [0]*14


DensCampo=[]
for n in range(len(nomes)): 
  DensCampo.append(ProprFil[n][' FieldDensity'])


#DENSI REL X DIST2fil-------------------------------------
cx=[0.5,1.,1.5]
cxErr=[0]*len(cx)
cy=[]
for n in range(len(nomes)):
  cy.append(GalsInFil_GradDensi[n].DensiRelGrad_Clean.values)
cyErr=[[0,0,0]]*14
# x_label='Distance to filament (Mpc)'
# y_label=r'$\rho_{fil}/\rho_{field}$'
# plota_CC_dist(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
# outname='DensiRel_Dist2Fil_compara.png'
# plt.savefig(outname)
# plt.close()

#DENSI REL X DIST2fil | individual-------------------------------------
fig,ax=plt.subplots(nrows=7, ncols=2,sharex=True,sharey=True)
for n in range(len(nomes)):
  axis=ax.flat[n]
  axis.errorbar(cx, cy[n], xerr=cxErr, yerr=cyErr[n], marker='o', capsize=2, mec='k', mfc=colors[n], \
  ms=6, elinewidth=1, ecolor=colors[n],color=colors[n])

  axis.set_title(nomes[n], fontsize=8, fontweight='bold', pad=-10)
  axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)

x_label=r'D$_{\perp}$ (Mpc)'
y_label=r'$\rho_{rel}$'
fig.set_size_inches(12.5, 8.5)
fig.text(0.43, 0.05, x_label, va='center')
fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')

outname='DensiRel_Dist2Fil_individual.png'
plt.savefig(outname)
plt.close()

#DENSI REL X DIST2fil | media-------------------------------------
ccy=[]
ccyErr=[]
for i in range(len(cx)):
  aux=[]
  for n in range(len(nomes)):
    aux.append(np.mean(cy[n][i]))
  ccy.append(np.mean(aux))
  ccyErr.append(np.std(aux)/np.sqrt(len(aux)))


plt.errorbar(cx, ccy, xerr=cxErr, yerr=ccyErr, marker='o', capsize=2, mec='k', mfc=colors[0], \
ms=6, elinewidth=1, ecolor=colors[0],linestyle=' ')
plt.tick_params(direction='in',top=True, right=True,labeltop=False, labelright=False)
x_label=r'D$_{\perp}$ (Mpc)'
y_label=r'$\rho_{rel}$'
plt.xlabel(x_label)
plt.ylabel(y_label)

cx=np.array(cx).ravel()
ccy = np.array(ccy).ravel()
ccyErr = np.array(ccyErr).ravel()


pars,cov=optimize.curve_fit(reta,cx,ccy, sigma=ccyErr)
errs=np.sqrt(np.diag(cov))
xfake=np.linspace(min(cx)-0.1,max(cx)+0.1,100)
yfit=reta(xfake,pars[0],pars[1])
yfitS=reta(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=reta(xfake,pars[0]-errs[0],pars[1]-errs[1])


plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='gray',alpha=0.5,edgecolor='w'
)
plt.xlim(0.4,1.6)
fit_lab=r'(%.2f $\pm$ %.2f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(0.45, 1.7, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))

outname='DensiRel_Dist2Fil_media.png'
plt.savefig(outname)
plt.close()

#DENSI REL X DIST2cl-------------------------------------
cx=[2.5,3.5,4.5,5.5]
cxErr=[0]*len(cx)
cy=[]
for n in range(len(nomes)):
  cy.append((DensiRel_slice[n]))
cyErr=[[0,0,0,0]]*14
# x_label='Distance to closest cluster (Mpc)'
# y_label=r'$\rho_{fil}/\rho_{field}$'
# plota_CC_dist(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
# outname='DensiRel_Dist2Cl_compara.png'
# plt.savefig(outname)
# plt.close()

#DENSI REL X DIST2cl | individual-------------------------------------
fig,ax=plt.subplots(nrows=7, ncols=2,sharex=True,sharey=True)
for n in range(len(nomes)):
  axis=ax.flat[n]
  axis.errorbar(cx, cy[n], xerr=cxErr, yerr=cyErr[n], marker='o', capsize=2, mec='k', mfc=colors[n], \
  ms=6, elinewidth=1, ecolor=colors[n],color=colors[n])

  axis.set_title(nomes[n], fontsize=8, fontweight='bold', pad=-10)
  axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)

x_label=r'D$_{agl}$ (Mpc)'
y_label=r'$\rho_{rel}$'
fig.set_size_inches(12.5, 8.5)
fig.text(0.5, 0.05, x_label, va='center')
fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')
outname='DensiRel_Dist2Cl_individual.png'
plt.savefig(outname)
plt.close()

#DENSI REL X DIST2cl | media-------------------------------------
ccy=[]
ccyErr=[]
for i in range(len(cx)):
  aux=[]
  for n in range(len(nomes)):
    aux.append(np.mean(cy[n][i]))
  ccy.append(np.mean(aux))
  ccyErr.append(np.std(aux)/np.sqrt(len(aux)))


plt.errorbar(cx, ccy, xerr=cxErr, yerr=ccyErr, marker='o', capsize=2, mec='k', mfc=colors[0], \
ms=6, elinewidth=1, ecolor=colors[0],linestyle=' ')
plt.tick_params(direction='in',top=True, right=True,labeltop=False, labelright=False)
x_label=r'D$_{agl}$ (Mpc)'
y_label=r'$\rho_{rel}$'
plt.xlabel(x_label)
plt.ylabel(y_label)

outname='DensiRel_Dist2Cl_media.png'
plt.savefig(outname)
plt.close()

#DENSI REL X z | DIST 2 FIL-------------------------------------
cx=f_agl[' redshift'].values
cxErr=[0]*len(cx)

# cy=ProprFil[' Lenght (Mpc)']

fig, ax=plt.subplots(nrows=2,ncols=2, sharex=True)
dist=[0,0.5,1.,1.5]
for i in range(4):
  cy=[]
  if i ==3: 
    for n in range(len(nomes)):
      cy.append(ProprFil[n][' Relative Density'])
      ylim=20
    tit='Dist2Fil < 1.5 Mpc'
  else:
    for n in range(len(nomes)):
      cy.append(GalsInFil_GradDensi[n].DensiRelGrad_Clean.values[i])
      ylim=4.2
    tit = str(dist[i]) + '< Dist2Fil <' + str(dist[i+1])
  cyErr=[0]*14
  axis=ax.flat[i]
  axis.set_ylim(0,ylim)
  axis.errorbar(cx, cy, xerr=cxErr, yerr=cyErr[n], marker='o', capsize=2, mec='k', mfc=colors[i], \
  ms=6, elinewidth=1, ecolor=colors[i],linestyle=' ')
  axis.set_title(tit, fontsize=8, fontweight='bold')
  axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
x_label=r'$z_{agl}$'
y_label=r'$\rho_{rel}$'
fig.text(0.5, 0.05, x_label, va='center')
fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')

outname='DensiRel_z_Dist2Fil.png'
plt.savefig(outname)
plt.close()

#LEN & Densi X z | -------------------------------------
cx=f_agl[' redshift'].values
cxErr=[0]*len(cx)

cyErr=[0]*14
legs=[]

fig,ax=plt.subplots(nrows=1,ncols=2, sharex=True)
for i in range(2):
  axis=ax.flat[i]
  if i ==0:
    cy = cxP
    y_label=r'$\rho_{rel}$'
  else:
    cy= cxL
    y_label=r'$L_{fil}$ (Mpc)'

  for n in range(len(nomes)):
    # yy=frac_red[n][coly]/frac_color[' RF_tot_CMP'][n]
    item=nomes[n]
    if (f_agl[' CC'][n] == 'NCC') :
      marca='o'
    elif (f_agl[' CC'][n] == 'CC') :
      marca='d'
    else :
        marca=''
    if f_agl['HighDens'][n] == 0:
      alp = 0.3
      meww=2.5
    else:
      alp=1
      meww=1

    axis.errorbar(cx[n], cy[n], xerr=cxErr[n], yerr=cyErr[n], marker=marca, capsize=2, mec='k', mfc=colors[n], \
  ms=6, elinewidth=1, ecolor=colors[n],alpha=alp,mew=meww)
    axis.set_ylabel(y_label)
    axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)

    if i==0:
      legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[n], mec='k', markersize=9, alpha=0.7, label=item))

legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor='k', markersize=7, alpha=0.7, label='NCC'))
legs.append(Line2D([0], [0], marker='d', color='w',markerfacecolor="k", markersize=7, alpha=0.7, label='CC'))

fig.set_size_inches(8.5,4.5)
fig.subplots_adjust(wspace=0.25, top=0.87)
fig.legend(handles=legs, loc='upper center', bbox_to_anchor=(0.51,1.),
          ncol=6, fancybox=True, shadow=True, fontsize = 'x-small')

x_label=r'$z_{agl}$'
fig.text(0.5, 0.05, x_label, va='center')


outname='DensiRel_Len_z_High'
plt.savefig(outname)
plt.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MTOT X PROPS FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#-----------------------------------------------
cy=f_agl[' Mtot_isoT(1e13Msun)'].values/10
cyErr=f_agl[' MtotErr'].values/10

## Mtot X DENSI-REL => TOT =================
cx=[]
cxErr=[]
cx2=[]
cx2Err=[]
cy2=[]
cy2Err=[]
cx3=[]
cx3Err=[]
cy3=[]
cy3Err=[]


for n in range(len(nomes)):
  cx.append(ProprFil[n][' Relative Density'].values[0])
  cxErr.append( (((ProprFil[n][' Relative DensityErrSup']).values +  ( ProprFil[n][' Relative DensityErrInf']))/2).values[0] )
  if f_agl['HighDens'][n] == 1:
    cx2.append(cx[n])
    cx2Err.append(cxErr[n])
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
  if (cy[n] < 6) and (f_agl['HighDens'][n] == 1):
    cx3.append(cx[n])
    cx3Err.append(cxErr[n])
    cy3.append(cy[n])
    cy3Err.append(cyErr[n])

y_label=r'M$_{500}$ ($10^{14} M_{\odot}$)'
x_label=r'$\rho_{rel}$'
plota_CC_HD(cx, cy, [0]*14, cyErr, f_agl, x_label, y_label)
#plt.title(r'$M_{agl}^T \times$ Relative density', fontsize=10)
plt.ylim(0.5,8.5)

cx=np.array(cx).ravel()
cxErr = np.array(cxErr).ravel()
cy = np.array(cy).ravel()
cyErr = np.array(cyErr).ravel()
cx2=np.array(cx2).ravel()
cx2Err = np.array(cx2Err).ravel()
cy2 = np.array(cy2).ravel()
cy2Err = np.array(cy2Err).ravel()
cx3=np.array(cx3).ravel()
cx3Err = np.array(cx3Err).ravel()
cy3 = np.array(cy3).ravel()
cy3Err = np.array(cy3Err).ravel()
xfake=np.linspace(min(cx)-1,max(cx)+1,100)

#AJUSTE ROBUSTO-------------------------------------------
linear=Model(reta2)
mydata = RealData(cx3, cy3,sy=cy3Err,sx=cx3Err)
myodr = ODR(mydata, linear, beta0=[1.,0.])
myoutput = myodr.run()
myoutput.pprint()
pars=myoutput.beta
errs=myoutput.sd_beta

xfake=np.linspace(min(cx)-1,max(cx)+1,100)
yfit=reta2(pars,xfake)
yfitS=reta2(pars+errs,xfake)
yfitI=reta2(pars-errs,xfake)

yfit=reta(xfake,pars[0],pars[1])
yfitS=reta2(pars+errs,xfake)
yfitI=reta2(pars-errs,xfake)

plt.plot(xfake,yfit,color='gray')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='dimgray',alpha=0.3,edgecolor='w'
)
plt.xlim(2.5,20)
fit_lab=r'(%.3f $\pm$ %.3f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(11, 1.1, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))


#REGRESSÃO LINEAR-----------------------------------
p,ers=RobustLinear(cx2,cy2,cy2Err)

yfit2=reta2(p,xfake)
yfit2S=reta(xfake,p1m+p1Err,p2m+p2Err)
yfit2I=reta(xfake,p1m-p1Err,p2m-p2Err)
plt.plot(xfake,yfit2,color='gray')
# plt.fill(
#     np.append(xfake, xfake[::-1]),
#     np.append(yfit2I, yfit2S[::-1]),
#     color='gray',alpha=0.5,edgecolor='w'
# )
plt.xlim(2,20)
plt.ylim(1,8)

plt.plot(xfake,yfit2, 'b')

fit_lab=r'(%.2f $\pm$ %.2f)X + (%.2f $\pm$ %.2f)' % (p1m,p1Err,p2m,p2Err)
plt.text(8.4, 1.6, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=1'))

#CURVE FIT-----------------------------------

pars,cov=optimize.curve_fit(reta,cx3,cy3, sigma=cy3Err)
errs=np.sqrt(np.diag(cov))
yfit=reta(xfake,pars[0],pars[1])
yfitS=reta(xfake,pars[0]+errs[0],pars[1]+errs[1])
yfitI=reta(xfake,pars[0]-errs[0],pars[1]-errs[1])

plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='gray',alpha=0.5,edgecolor='w'
)

outname='DensiRel_Mtot_NOFIT.png'
plt.savefig(outname)
plt.close()

## Mtot X DENSI-REL => closest to Cluster | FATIA =================
## Mtot X LEN  =================
cx=[]
cxErr=[]
cx2=[]
cy2=[]
cy2Err=[]
cx2Err=[]
for n in range(len(nomes)):
  cx.append(ProprFil[n][' Length (Mpc)'].values[0])
  cxErr.append( (((ProprFil[n][' LengthErrSup']).values +  ( ProprFil[n][' LengthErrInf']))/2).values[0] )

  if f_agl['HighDens'][n] == 1:
    cx2.append(cx[n])
    cy2.append(cy[n])
    cy2Err.append(cyErr[n])
    cx2Err.append(cxErr[n])
cxErr=[0]*14

y_label=r'M$_{500}$ ($10^{14} M_{\odot}$)'
x_label=r'$L_{fil}$ (Mpc)'
plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)


#REGRESSÃO LINEAR-----------------------------------
def f2(p,x):
  a,b=p
  func=a/(x) + b
  return func

pars,errs=fitReta(cx2,cy2,cx2Err,cy2Err)

linear=Model(reta2)
mydata = RealData(cx2, cy2,sy=cy2Err)
myodr = ODR(mydata, linear, beta0=[-1.,1])
myoutput = myodr.run()
myoutput.pprint()
pars=myoutput.beta
errs=myoutput.sd_beta

xfake=np.linspace(min(cx)-5,max(cx)+10,100)
yfit=f2(pars,cx)
yfitS=f2(pars+errs,xfake)
yfitI=f2(pars-errs,xfake)

plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='gray',alpha=0.5,edgecolor='w'
)

plt.xlim(10,180)
plt.ylim(1,9)

fit_lab=r' $\frac{(%i \pm %i)}{X}$ + (%.1f $\pm$ %.1f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(100,8.1, fit_lab, va='center',\
  bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=1', alpha=0.5))

outname='Len_Mtot_NOFIT.png'
plt.savefig(outname)
plt.close()

## Mtot X DENSIREL => closest to FIL  =================
fig, ax=plt.subplots(nrows=1,ncols=3, sharex=True,sharey=True)
dists=[0.,0.5,1.,1.5]
for i in range(3):
  cx=[]
  cx2=[]
  for n in range(len(nomes)):
    cx.append(DensiRel_grad[n][i])
    if f_agl['HighDens'][n] == 1:
      cx2.append(cx[n])
  cxErr=[0]*14
  tit=str(dists[i]) +'< Dist2Fil <' + str(dists[i+1])
  axis=ax.flat[i]
  axis.errorbar(cx2, cy2, xerr=cx2Err, yerr=cy2Err, marker='o', capsize=2, mec='k', mfc=colors[i], \
  ms=6, elinewidth=1, ecolor=colors[i],linestyle=' ')
  axis.set_title(tit, fontsize=8, fontweight='bold')
  axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
x_label=r'$\rho_{fil}/\rho_{field}$'
y_label=r'M$_{agl}^{T}$ ($10^{14} M_{\odot}$)'
fig.text(0.5, 0.05, x_label, va='center')
fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')
fig.set_size_inches(12.5,4.5)
outname='DensiRel_Mtot_Dist2Fil_HighDens.png'
plt.savefig(outname)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# TEMP X PROPS FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cy=f_agl[' kTm'].values
cyErr=((f_agl[' kTm+'].values - cx) + (cx - f_agl[' kTm-'].values ))/2

## TEMP X DENSI-REL => closest to Cluster | FATIA =================

cx=DensiRel_3_1
cxErr=[0]*14
y_label=r'kT$_{agl} (keV)$'
x_label=r'$\rho_{fil}/\rho_{field}$'
plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plt.title(r'$T_{agl} \times$ Relative density', fontsize=10)
outname='DensiRel_kT_innerCl3.png'
plt.savefig(outname)
plt.close()


## TEMP X LEN => closest to Cluster | FATIA =================
cx=[]
for n in range(len(nomes)):
  cx.append(ProprFil[n][' Length (Mpc)'].values[0])
cxErr=[0]*14
y_label=r'kT$_{agl} (keV)$'
x_label='Filament Length (Mpc)'
plota_CC(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)
plt.title(r'$T_{agl} \times$ Filament lenght', fontsize=10)
outname='Len_kT.png'
plt.savefig(outname)
plt.close()


## TEMP X DENSIREL => closest to FIL  =================
fig, ax=plt.subplots(nrows=1,ncols=3, sharex=True,sharey=True)
dists=[0.,0.5,1.,1.5]
for i in range(3):
  cx=[]
  for n in range(len(nomes)):
    cx.append(DensiRel_grad[n][i])
  cxErr=[0]*14
  tit=str(dists[i]) +'< Dist2Fil <' + str(dists[i+1])
  axis=ax.flat[i]
  axis.errorbar(cx, cy, xerr=cxErr, yerr=cyErr, marker='o', capsize=2, mec='k', mfc=colors[i], \
  ms=6, elinewidth=1, ecolor=colors[i],linestyle=' ')
  axis.set_title(tit, fontsize=8, fontweight='bold')
  axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
x_label=r'$\rho_{fil}/\rho_{field}$'
y_label=r'kT$_{agl} (keV)$'
fig.text(0.5, 0.05, x_label, va='center')
fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')
fig.set_size_inches(12.5,4.5)

outname='DensiRel_kT_Dist2Fil.png'
plt.savefig(outname)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# COR AGL X PROPS FIL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cy=[]
cyErr=[]
for n in range(len(nomes)):
  cy.append(np.mean(GalsInCl[n]['gi']))
  cyErr.append(np.std(GalsInCl[n]['gi'])/np.sqrt(len(GalsInCl[n]['gi'])))

## COR AGL X DENSI-REL => closest to Cluster | INNER =================

cx=DensiRel_3_1
cxErr=[0]*14
y_label=r'$(g-i)_{agl}$'
x_label=r'$\rho_{fil}/\rho_{field}$'
plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)


md=pd.DataFrame({'densi':cx, 'gi':cy, 'gierr': cyErr})
bini=md.groupby(pd.cut(md.densi,3)).mean()
binierr=md.groupby(pd.cut(md.densi,3)).std()
ct=md.densi.value_counts(bins=3).values
plt.errorbar(bini.densi,bini.gi,xerr=binierr.densi/np.sqrt(ct), yerr=bini.gierr, color='gray',\
 marker='*', ms=10,linestyle=' ',capsize=5, mec='k', mfc='gray', elinewidth=1, ecolor='gray')
plt.title(r'$(g-i)_{agl} \times$ Relative density, Dist2Agl <=3', fontsize=10)
outname='DensiRel_giAgl_innerCl3.png'
plt.savefig(outname)
plt.close()

## COR AGL X LEN => closest to Cluster | FATIA =================
cx=[]
for n in range(len(nomes)):
  cx.append(ProprFil[n][' Length (Mpc)'].values[0])
cxErr=[0]*14
y_label=r'$(g-i)_{agl}$'
x_label='Filament Length (Mpc)'
plota_CC_HD(cx, cy, cxErr, cyErr, f_agl, x_label, y_label)

b=3
md=pd.DataFrame({'len':cx, 'gi':cy, 'gierr': cyErr})
bini=md.groupby(pd.cut(md.len,b)).mean()
binierr=md.groupby(pd.cut(md.len,b)).std()
ct=md.len.value_counts(bins=b).values
plt.errorbar(bini.len,bini.gi,xerr=binierr.len/np.sqrt(ct), yerr=bini.gierr, color='gray',\
 marker='*', ms=10,linestyle=' ',capsize=5, mec='k', mfc='gray', elinewidth=1, ecolor='gray')
plt.title(r'$(g-i)_{agl} \times$ Filament lenght', fontsize=10)
outname='Len_giAgl.png'
plt.savefig(outname)
plt.close()

## COR AGL X DENSIREL => closest to FIL  =================
fig, ax=plt.subplots(nrows=1,ncols=3, sharex=True,sharey=True)
dists=[0.,0.5,1.,1.5]
for i in range(3):
  cx=[]
  for n in range(len(nomes)):
    cx.append(DensiRel_grad[n][i])
  cxErr=[0]*14
  tit=str(dists[i]) +'< Dist2Fil <' + str(dists[i+1])
  axis=ax.flat[i]
  axis.errorbar(cx, cy, xerr=cxErr, yerr=cyErr, marker='o', capsize=2, mec='k', mfc=colors[i], \
  ms=6, elinewidth=1, ecolor=colors[i],linestyle=' ')
  axis.set_title(tit, fontsize=8, fontweight='bold')
  axis.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
x_label=r'$\rho_{fil}/\rho_{field}$'
y_label=r'$(g-i)_{agl}$'
fig.text(0.5, 0.05, x_label, va='center')
fig.text(0.07, 0.5, y_label, va='center', rotation='vertical')
fig.set_size_inches(12.5,4.5)
outname='DensiRel_giAgl_Dist2Fil.png'
plt.savefig(outname)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MAPAS DENSIDADE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#MAPA DE DENSIDADE gi X 'Dist2Cluster (Mpc)'--------------------------
#dist = [0,0.5]

fig,ax=plt.subplots(nrows=7, ncols=2, sharey=True, sharex=True)
n=0
for axis in ax.flat:
  dd=GalsInFil_clean02[n].query('Dist2ClosCluster_Mpc < 5.')
  xx=dd.Dist2ClosCluster_Mpc.values
  yy=dd.gi.values#ri == gi, pq ta errado na table original
  xyrange = [[min(xx),max(xx)],[min(yy), max(yy)]]
  bins=[20,10]
  hh, locx, locy = scipy.histogram2d(xx, yy, range=xyrange, bins=bins)
  # im=axis.imshow(np.flipud(hh.T),cmap='hot_r',extent=np.array(xyrange).flatten(), interpolation='none', origin='upper', aspect='auto')
  im=axis.imshow(np.flipud(hh.T),cmap='gnuplot2_r',extent=np.array(xyrange).flatten(), interpolation='none', origin='upper', aspect='auto')
  axis.set_title(nomes[n], fontsize=8, fontweight='bold')
  axis.tick_params(direction='in',labelsize=8)
  n+=1

ylab=r'$g - i$'
xlab='Distance from closest cluster (Mpc)'
fig.text(0.02, 0.5, ylab, va='center', rotation='vertical', size=12)
fig.text(0.4, 0.01, xlab, va='center', size=12)
plt.subplots_adjust(left  = 0.06,right = 0.93,bottom = 0.05,top = 0.95,wspace = 0.07,hspace = 0.395)
cbar_ax = fig.add_axes([0.95, 0.05, 0.02, 0.9])
fig.colorbar(im, cax=cbar_ax)
fig.set_size_inches(12.5, 8.5)
plt.savefig('gi_Dist2Cl_DensiMap_5.png')
plt.close()

#MAPA DE DENSIDADE gi X 'Dist2Fil (Mpc)'--------------------------
#dist = [0,0.5]

fig,ax=plt.subplots(nrows=7, ncols=2, sharey=True, sharex=True)
n=0
for axis in ax.flat:
  dd=GalsInFil_clean02[n]
  xx=dd.Dist2Fil_Mpc.values
  yy=dd.gi.values#ri == gi, pq ta errado na table original
  xyrange = [[min(xx),max(xx)],[min(yy), max(yy)]]
  bins=[15,3]
  hh, locx, locy = scipy.histogram2d(xx, yy, range=xyrange, bins=bins)
  # im=axis.imshow(np.flipud(hh.T),cmap='hot_r',extent=np.array(xyrange).flatten(), interpolation='none', origin='upper', aspect='auto')
  im=axis.imshow(np.flipud(hh.T),cmap='gnuplot2_r',extent=np.array(xyrange).flatten(), interpolation='none', origin='upper', aspect='auto')
  axis.set_title(nomes[n], fontsize=8, fontweight='bold')
  axis.tick_params(direction='in',labelsize=8)
  n+=1

ylab=r'$g - i$'
xlab='Distance from closest cluster (Mpc)'
fig.text(0.02, 0.5, ylab, va='center', rotation='vertical', size=12)
fig.text(0.4, 0.01, xlab, va='center', size=12)
plt.subplots_adjust(left  = 0.06,right = 0.93,bottom = 0.05,top = 0.95,wspace = 0.07,hspace = 0.395)
cbar_ax = fig.add_axes([0.95, 0.05, 0.02, 0.9])
fig.colorbar(im, cax=cbar_ax)
fig.set_size_inches(12.5, 8.5)
plt.savefig('gi_Dist2Cl_DensiMap_5.png')
plt.close()

#HISTOGRAMA pontos X dist2cluster----------------------
fig,ax=plt.subplots(nrows=7, ncols=2,sharex=True, sharey=True)
n=0
for axis in ax.flat:
  dd=GalsInFil_clean[n].query('1.5 < Dist2ClosCluster_Mpc < 5.')
  xx=dd.Dist2ClosCluster_Mpc.values
  weights = np.ones_like(xx)/float(len(xx))
  axis.hist(xx, bins=4,density=True, color='lightcoral', alpha=0.9, rwidth=0.8, weights=weights)
  axis.tick_params(direction='in',labelsize=8)
  axis.set_title(nomes[n], fontsize=8, fontweight='bold')
  # axis.set_xlim(left=min(xx), right=max(xx))
  n+=1
ylab=r'$g - i$'
xlab='Distance from closest cluster (Mpc)'
fig.text(0.42, 0.01, xlab, va='center', size=12)
plt.subplots_adjust(left  = 0.06,right = 0.93,bottom = 0.05,top = 0.95,wspace = 0.07,hspace = 0.395)
fig.set_size_inches(12.5, 8.5)
plt.savefig('Dist2Cl_hist_5.png')
plt.close()


#gi X dist2cluster ( distribuição de ptos)-----------------
# < 5 mpc
fig,ax=plt.subplots(nrows=7, ncols=2,sharex=True, sharey=True)
n=0
for axis in ax.flat:
  dd=GalsInFil_clean[n].query('1.5 < Dist2ClosCluster_Mpc < 5.')
  xx=dd.Dist2ClosCluster_Mpc.values
  yy=dd.gi.values
  axis.scatter(xx,yy, marker='o', color=colors[n], edgecolor='k')
  axis.tick_params(direction='in',labelsize=8)
  axis.set_title(nomes[n], fontsize=8, fontweight='bold')

  # axis.tick_params(direction='in')
  n+=1
ylab=r'$g - i$'
xlab='Distance from closest cluster (Mpc)'
fig.text(0.02, 0.5, ylab, va='center', rotation='vertical', size=12)
fig.text(0.42, 0.01, xlab, va='center', size=12)
plt.subplots_adjust(left  = 0.06,right = 0.98,bottom = 0.05,top = 0.95,wspace = 0.07,hspace = 0.395)
fig.set_size_inches(12.5, 8.5)
plt.savefig('gi_Dist2Cl_5.png')
plt.close()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FRAÇÃO DE AZUIS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# gi_azuis e gi_vermelhas x dist2cluster------------------------------------------
azul=[]
verm=[]
azulCl=[]
vermCl=[]
azulCl_drt=[]
vermCl_drt=[]
for n in range(len(nomes)):
  ang=RSfit[n].Ang_RS_cluster.values[0]
  lin=RSfit[n].ZeroPoint_RS_cluster.values[0]
  azul.append(GalsInFil_clean02[n].query( 'gr < (@ang*Mr + @lin -0.15)' ))
  verm.append(GalsInFil_clean02[n].query( 'gr >= (@ang*Mr + @lin -0.15)' ))
  azulCl.append(GalsInCl[n].query( 'gr < (@ang*Mr + @lin -0.15)' ))
  vermCl.append(GalsInCl[n].query( 'gr >= (@ang*Mr + @lin -0.15)' ))
  azulCl_drt.append(GalsInCl_drt[n].query( 'gr < (@ang*Mr + @lin -0.15)' ))
  vermCl_drt.append(GalsInCl_drt[n].query( 'gr >= (@ang*Mr + @lin -0.15)' ))


fig,ax=plt.subplots(nrows=7, ncols=2,sharex=True, sharey=True)
n=0
for axis in ax.flat:
  # d1=verm[n].query('1.5 < Dist2ClosCluster_Mpc < 8.5')
  # d2=azul[n].query('1.5 < Dist2ClosCluster_Mpc < 8.5')
  d1=verm[n].query('Dist2ClosCluster_Mpc < 8.5')
  d2=azul[n].query('Dist2ClosCluster_Mpc < 8.5')
  xx1=d1.Dist2ClosCluster_Mpc.values
  xx2=d2.Dist2ClosCluster_Mpc.values
  binwidth=1
  axis.hist(xx1, bins=range(int(min(xx1)), int(max(xx1)) + binwidth, binwidth),density=True, color=colors[5], alpha=1, rwidth=0.8, edgecolor=colors[5], linewidth=1.2)
  axis.hist(xx2, bins=range(int(min(xx1)), int(max(xx1)) + binwidth, binwidth),density=True, color=colors[1], alpha=0.6, rwidth=0.8, edgecolor=colors[1], linewidth=1.2)
  axis.tick_params(direction='in',labelsize=8)
  axis.set_title(nomes[n], fontsize=8, fontweight='bold')
  # axis.set_xlim(left=min(xx), right=max(xx))
  n+=1
ylab=r'$g - i$'
xlab='Distance from closest cluster (Mpc)'
fig.text(0.42, 0.01, xlab, va='center', size=12)
plt.subplots_adjust(left  = 0.06,right = 0.93,bottom = 0.05,top = 0.95,wspace = 0.07,hspace = 0.395)
fig.set_size_inches(12.5, 8.5)
plt.savefig('Dist2Cl_hist_cores.png')
plt.close()

# Fração de azuis FILAMENTO X redshift -----------------------------------------
cx=f_agl[' redshift']
cxErr=[0]*14
cy=[]
cyErr=[]
for n in range(len(nomes)):
  cy.append(len(azul[n].gi)/(len(azul[n].gi) + len(verm[n].gi)))
  cyErr.append(0)

ylab=r'$BF_{fil}$'
xlab=r'$z_{agl}$'

plota_CC_HD(cx,cy,cxErr,cyErr,f_agl,xlab,ylab)
plt.ylim(-0.02,0.4)

cx=np.array(cx.values).ravel()
cy = np.array(cy).ravel()

linear=Model(reta2)
mydata = RealData(cx, cy)
myodr = ODR(mydata, linear, beta0=[1.,0.])
myoutput = myodr.run()
myoutput.pprint()
pars=myoutput.beta
errs=myoutput.sd_beta

xfake=np.linspace(min(cx)-0.1,max(cx)+0.1,100)
yfit=reta2(pars,xfake)
yfitS=reta2(pars+errs,xfake)
yfitI=reta2(pars-errs,xfake)

plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='gray',alpha=0.5,edgecolor='w'
)
plt.xlim(0.12,0.38)
fit_lab=r'(%.2f $\pm$ %.2f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(0.14, 0.02, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))


outname='z_BF_fil.png'
plt.savefig(outname)
plt.close()

# Fração de azuis CLUSTER X redshift -----------------------------------------
cx=f_agl[' redshift']
cxErr=[0]*14
cy=[]
cyErr=[]
for n in range(len(nomes)):
  cy.append(len(azulCl[n].gi)/(len(azulCl[n].gi) + len(vermCl[n].gi)))
  cyErr.append(0.0001)

ylab=r'$BF_{agl}$'
xlab=r'$z_{agl}$'

# pars,errs=optimize.curve_fit(reta,cx,cy)

plota_status(cx,cy,cxErr,cyErr,f_agl,xlab,ylab)
plt.ylim(-0.02,0.4)
cx=np.array(cx.values).ravel()
cy = np.array(cy).ravel()
cxErr=np.array(cxErr.values).ravel()
cyErr = np.array(cyErr).ravel()
# pars,errs=fitReta(cx,cy,cxErr,cyErr)


linear=Model(reta2)
mydata = RealData(cx, cy)
myodr = ODR(mydata, linear, beta0=[1.,0.])
myoutput = myodr.run()
myoutput.pprint()
pars=myoutput.beta
errs=myoutput.sd_beta

xfake=np.linspace(min(cx)-0.1,max(cx)+0.1,100)
yfit=reta2(pars,xfake)
yfitS=reta2(pars+errs,xfake)
yfitI=reta2(pars-errs,xfake)

plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='gray',alpha=0.5,edgecolor='w'
)
plt.xlim(0.12,0.38)
fit_lab=r'(%.2f $\pm$ %.2f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(0.14, 0.36, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))

outname='z_BF_Cl.png'
plt.savefig(outname)
plt.close()

#BFFil X Dist2Fil-------------------------------------------------

cyErr=[]
for n in range(len(nomes)):
  cy.append(len(azul[n].gi)/(len(azul[n].gi) + len(verm[n].gi)))
  cyErr.append(0)

cy=[]
cyErr=[]
dists=[0,0.5,1.,1.5]
for i in range(len(dists)-1):
  auxi=[]
  auxiEr=[]
  for n in range(len(nomes)):
    azul1=azul[n].loc[(dists[i] < azul[n]['Dist2Fil_Mpc'])]
    azul2= azul1.loc[azul1['Dist2Fil_Mpc']<= dists[i+1]]

    verm1=verm[n].loc[(dists[i] < verm[n]['Dist2Fil_Mpc'])]
    verm2= verm1.loc[verm1['Dist2Fil_Mpc']<= dists[i+1]]
    auxi.append(len(azul2.gi)/(len(azul2.gi) + len(verm2.gi)))
  cy.append(np.mean(auxi))
  cyErr.append(np.std(auxi)/np.sqrt(len(auxi)))

cx=[0.5,1.,1.5]
cxErr=[0,0,0]

plt.errorbar(cx, cy, xerr=cxErr, yerr=cyErr, marker='o', capsize=2, mec='k', mfc=colors[0], \
ms=6, elinewidth=1, ecolor=colors[0],linestyle=' ')

plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
x_label='Distance to filament (Mpc)'
y_label='BF'
plt.xlabel(x_label)
plt.ylabel(y_label)
outname='BF_Dist2Fil.png'
plt.savefig(outname)
plt.close()

#CDF E PDF PELA DISTANCIA AO CLUSTER-------------------------------------------------
#divide entre lowz e highz
col=azul[0].columns
lowZ_azul=pd.DataFrame(columns=col)
lowZ_verm=pd.DataFrame(columns=col)
hiZ_azul=pd.DataFrame(columns=col)
hiZ_verm=pd.DataFrame(columns=col)
for n in range(len(nomes)):
  if f_agl[' redshift'][n] < 0.25:
    lowZ_azul=pd.concat([lowZ_azul,azul[n]])
    lowZ_verm=pd.concat([lowZ_verm,verm[n]])
  else:
    hiZ_azul=pd.concat([hiZ_azul,azul[n]])
    hiZ_verm=pd.concat([hiZ_verm,verm[n]])



Alow,Ahi,p_A=KS_CDF(lowZ_azul.query('Dist2ClosCluster_Mpc < 3').Dist2ClosCluster_Mpc , hiZ_azul.query('Dist2ClosCluster_Mpc < 3').Dist2ClosCluster_Mpc)
lablow=r'$z<0.25$'
labhi=r'$z>0.25$'
plot_2CDF(Alow[0],Alow[1],Ahi[0],Ahi[1],lablow,labhi)


Vlow,Vhi,p_V=KS_CDF(lowZ_verm.query('Dist2ClosCluster_Mpc < 3').Dist2ClosCluster_Mpc , hiZ_verm.query('Dist2ClosCluster_Mpc < 3').Dist2ClosCluster_Mpc)
plot_2CDF(Ahi[0],Ahi[1],Vhi[0],Vhi[1],lablow,labhi)


def CDF(x):
  xx=np.random.choice(x,size=len(x),replace=True)
  H1,X1=np.histogram(xx , bins = 30, density=True)
  dx = X1[1] - X1[0]
  F1 = np.cumsum(H1)*dx
  return F1,X1


def KS_CDF(A,V):
  p=[]
  FAA=[]
  FVV=[]
  for _ in range(1000):
    xA=np.random.choice(A,size=len(A),replace=True)
    HA,XA=np.histogram(xA , bins = 10, density=True)
    dxA = XA[1] - XA[0]
    FA = np.cumsum(HA)*dxA
    FAA.append(FA)

    xV=np.random.choice(V,size=len(V),replace=True)
    HV,XV=np.histogram(xV , bins = 10, density=True)
    dxV = XV[1] - XV[0]
    FV = np.cumsum(HV)*dxV
    FVV.append(FV)

    KS, p_cdf = stats.ks_2samp(FA, FV)
    p.append(p_cdf)

  XA_m=XA
  XV_m=XV
  FA_mean=[]
  FA_err=[]
  FV_mean=[]
  FV_err=[]
  for i in range(len(FAA[0])):
    print(i)
    aux_mA=[]
    aux_mV=[]
    for n in range(1000):
      aux_mA.append(FAA[n][i])
      aux_mV.append(FVV[n][i])
    FA_mean.append(np.mean(aux_mA))
    FA_err.append(np.std(aux_mA)/np.sqrt(len(aux_mA)))
    FV_mean.append(np.mean(aux_mV))
    FV_err.append(np.std(aux_mV)/np.sqrt(len(aux_mV)))

  return [FA_mean,XA_m,FA_err],[FV_mean,XV_m,FV_err],[np.mean(p),(np.std(p)/np.sqrt(len(p)))]

def plot_2CDF(F1,X1,F2,X2,lab1,lab2):
  # formatter = FuncFormatter(log_10_product)

  plt.plot(X1[1:], F1, color='royalblue', label = lab1)
  plt.plot(X2[1:], F2, color='lightcoral', label = lab2)
  plt.xlabel('Galaxy-filament distance (MPC)')
  plt.ylabel('CDF')
  plt.legend(loc='best',fancybox=True, framealpha=0.3, fontsize='medium')
  plt.xscale('log')
  plt.show()
  return 




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONECTIVIDADE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cyInner=[]
cyOuter=[]
cyTot=[]
for n in range(len(nomes)):
  cyInner.append(Connect.FPInner[n])
  cyOuter.append(Connect.FPouter[n])
  cyTot.append(Connect.FPInner[n]+Connect.FPouter[n] +Connect.FP[n])
cyErr=[0]*14
cy1=cyInner+Connect.FP.values

cy15=[]
cy8=[]
cy8Tot=[]
for n in range(len(nomes)):
  cy15.append(Connectf.R15[n])
  cy8.append(Connectf.R8[n])
  cy8Tot.append(Connectf.R15[n]+Connectf.R8[n])
cyErr=[0]*14

# CONNECT X MASSA REFS-------------------------
cx=f_agl[' Mtot_isoT(1e13Msun)'].values*1e13
cxErr=f_agl[' MtotErr'].values*1e13


cx2=M200(cx)
cx2Err=M200(cxErr)

ylab=r'$\kappa$ (r < 8 Mpc h$^{-1})$'
xlab=r'log$_{10}$(M$_{200}$/M$_{\odot})$'

# plota_status(cx,cyInner+Connect.FP.values,cxErr,cyErr,f_agl,xlab,ylab)

#media em bins de massa
# md=pd.DataFrame({'M':cx2, 'Merr': cx2Err, 'C':cyInner+Connect.FP.values})
nb=4
md=pd.DataFrame({'M':cx2, 'Merr': cx2Err, 'C':cy8Tot})
bini=md.groupby(pd.cut(md.M,nb)).mean()
binierr=md.groupby(pd.cut(md.M,nb)).std()
ct=md.M.value_counts(bins=nb).values

xx=np.log10(bini.M.values)
yy=bini.C.values
# xxEr=binierr.Merr.values
# xxErr=np.log10(binierr.Merr.values)
xxEr=[0]*nb
yyEr=(binierr.C/np.sqrt(ct)).values

plt.errorbar(xx,yy, xerr=xxEr,yerr=yyEr, marker='o', capsize=2, mec='k', mfc=colors[0], \
  ms=6, elinewidth=1, ecolor=colors[0],linestyle=' ')

refs=pd.read_csv('/home/natalia/Desktop/aragon2010.csv',delimiter=',')
#Plota outros trabalho
for n in range(len(refs)):
  xr=refs['log(M/Ms)'][n]
  yr=refs['connect'][n]
  yrErr=refs['connectErr'][n]
  cc=refs['pp'][n]
  plt.errorbar(xr,yr,yerr=yrErr, marker='o', capsize=2, mec='k', mfc=colors[cc+2], \
  ms=6, elinewidth=1, ecolor=colors[cc+2],linestyle=' ')

legs=[]
legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor=colors[0], markersize=7, alpha=0.7, label='Este trabalho'))
legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor=colors[1+2], markersize=7, alpha=0.7, label='Aragon+10'))
legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor=colors[2+2], markersize=7, alpha=0.7, label='Malavasi+19'))
legs.append(Line2D([0], [0], marker='o', color='w',markerfacecolor=colors[3+2], markersize=7, alpha=0.7, label='Sarron+19'))
plt.legend(handles=legs, loc='upper left',
        ncol=1, fancybox=True, shadow=True, fontsize = 'x-small')

# xxEr=[0.1]*3
# xfake=np.linspace(min(xx)-1,max(xx)+1,100)
# pars,errs=fitReta(xx,yy,xxEr,yyEr)
# pars,errs=optimize.curve_fit(reta,xx,yy,sigma=yyEr)
# yfit=reta2(pars,xfake)
# yfitS=reta2(pars+errs,xfake)
# yfitI=reta2(pars-errs,xfake)

# plt.plot(xfake,yfit,color='k')
# plt.fill(
#     np.append(xfake, xfake[::-1]),
#     np.append(yfitI, yfitS[::-1]),
#     color='gray',alpha=0.5,edgecolor='w'
# )
# fit_lab=r'(%.2f $\pm$ %.2f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
# plt.text(2.7, 3.75, fit_lab, va='center',color='k',\
#   bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))

plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
plt.ylim(0,6.8)
plt.xlim(14,15.05)
plt.xlabel(xlab)
plt.ylabel(ylab)
outname='Connect_Mtot_refs.png'
plt.savefig(outname)
plt.close()



# CONNECT X MASSA -------------------------
cx=f_agl[' Mtot_isoT(1e13Msun)'].values/10
cxErr=f_agl[' MtotErr'].values/10


ylab=r'$\kappa$ (r < 8 Mpc h$^{-1})$'
xlab=r'M$_{agl}^T $ (10$^{14}$ M$_{\odot})$'

# plota_status(cx,cyInner+Connect.FP.values,cxErr,cyErr,f_agl,xlab,ylab)

#media em bins de massa
# md=pd.DataFrame({'M':cx2, 'Merr': cx2Err, 'C':cyInner+Connect.FP.values})
nb=4
md=pd.DataFrame({'M':cx, 'Merr': cxErr, 'C':cy8Tot})
bini=md.groupby(pd.cut(md.M,nb)).mean()
binierr=md.groupby(pd.cut(md.M,nb)).std()
ct=md.M.value_counts(bins=nb).values

xx=np.log10(bini.M.values)
yy=bini.C.values
# xxEr=binierr.Merr.values
# xxErr=np.log10(binierr.Merr.values)
xxEr=[0]*nb
yyEr=(binierr.C/np.sqrt(ct)).values

plt.errorbar(xx,yy, xerr=xxEr,yerr=yyEr, marker='o', capsize=2, mec='k', mfc=colors[0], \
  ms=6, elinewidth=1, ecolor=colors[0],linestyle=' ')

xxEr=[0.1]*nb
xfake=np.linspace(min(xx)-1,max(xx)+1,100)
pars,errs=fitReta(xx,yy,xxEr,yyEr)
# pars,errs=optimize.curve_fit(reta,xx,yy,sigma=yyEr)
yfit=reta2(pars,xfake)
yfitS=reta2(pars+errs,xfake)
yfitI=reta2(pars-errs,xfake)

plt.plot(xfake,yfit,color='k')
plt.fill(
    np.append(xfake, xfake[::-1]),
    np.append(yfitI, yfitS[::-1]),
    color='gray',alpha=0.5,edgecolor='w'
)
fit_lab=r'(%.2f $\pm$ %.2f)X + (%.2f $\pm$ %.2f)' % (pars[0],errs[0],pars[1],errs[1])
plt.text(0.33, 2.3, fit_lab, va='center',color='k',\
  bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=1'))

plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
plt.ylim(2,5)
plt.xlim(0.3,1)
plt.xlabel(xlab)
plt.ylabel(ylab)
outname='Connect_Mtot.png'
plt.savefig(outname)
plt.close()
# CONNECT X Z -------------------------
cx=f_agl[' redshift'].values
cxErr=[0]*14

plota_status(cx,cy8Tot,cxErr,cyErr,f_agl,xlab,ylab)

#====================================
#plota estado dinamico
#====================================
#exclui mergers
dfx=f_agl
colors = cm.tab20(np.linspace(0, 1, len(dfx.index.values)))


# ax1=plt.subplot(2,3,1)
# ax2=plt.subplot(2,3,2)
# ax3=plt.subplot(2,3,3)
# ax4=plt.subplot(2,3,4)
# ax5=plt.subplot(2,3,5)
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,2,3)
ax4=plt.subplot(2,2,4)

fig,ax=plt.subplots(nrows=1, ncols=3)

n_exclud=['A1914','A1758', 'ZWCL2341.1+0000']

legs=[]
axis=ax.flat
for d in range(len(f_agl)):
  if  nomes[d] not in n_exclud:
    item = nomes[d]
    col=' Cuspiness'
    col_er=' Cuspiness_err'
    lim=0.46
    ax1=axis[0]
    ax1.errorbar(f_agl[col][d], f_agl[col][d], xerr=f_agl[col_er][d], yerr=f_agl[col_er][d], color=colors[d], marker='.', ms=8, mec='k', capsize=2, mfc=colors[d])
    ax1.axvline(lim, linestyle='--', color='grey',linewidth=0.5)
    ax1.axhline(lim, linestyle='--', color='grey',linewidth=0.5)
    ax1.fill_between([lim,1],lim,0.9, facecolor='dimgray', alpha=0.05)
    ax1.set_ylim(-0.10,0.9)
    ax1.set_xlim(-0.10,0.9)
    # ax1.set_title('Cuspiness')
    ax1.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
    ax1.set_xlabel(r'$\delta$')
    ax1.set_ylabel(r'$\delta$')


    col=' CSB'
    col_er=' CSB_err'
    lim=0.26
    ax2=axis[1]
    ax2.errorbar(f_agl[col][d], f_agl[col][d], xerr=f_agl[col_er][d], yerr=f_agl[col_er][d]*1, color=colors[d], marker='.', ms=8, mec='k', capsize=2, mfc=colors[d])
    # ax2.errorbar(f_agl[col][d], f_agl[col][d], xerr=f_agl[col_er][d], yerr=f_agl[col_er][d], color=colors[d], marker='.', ms=8, mec='k', capsize=2, mfc=colors[d])
    ax2.axvline(lim, linestyle='--', color='grey',linewidth=0.5)
    ax2.axhline(lim, linestyle='--', color='grey',linewidth=0.5)
    ax2.fill_between([lim,0.4],lim,0.4, facecolor='dimgray', alpha=0.05)
    ax2.set_ylim(0,0.4)
    ax2.set_xlim(0,0.4)
    # ax2.set_title('CSB')
    ax2.set_xlabel(r'$CSB$')
    ax2.set_ylabel(r'$CSB$')

    # col=' CSB4'
    # col_er=' CSB4_err'
    # lim=0.055
    # ax4=axis[1]
    # ax4.errorbar(f_agl[col][d], f_agl[col][d], xerr=f_agl[col_er][d], yerr=f_agl[col_er][d], color=colors[d], marker='.', ms=8, mec='k', capsize=2, mfc=colors[d])
    # ax4.axvline(lim, linestyle='--', color='grey',linewidth=0.5)
    # ax4.axhline(lim, linestyle='--', color='grey',linewidth=0.5)
    # ax4.fill_between([lim,0.105],lim,0.105, facecolor='gray', alpha=0.4)
    # ax4.set_ylim(0.005,0.105)
    # ax4.set_xlim(0.005,0.105)
    # ax4.set_title('CSB4')

    col=' kT_ratio'
    col_er=' kT_ratioErr'
    lim=1.
    ax3=axis[2]
    ax3.errorbar(f_agl[col][d], f_agl[col][d], xerr=f_agl[col_er][d], yerr=f_agl[col_er][d], color=colors[d], marker='.', ms=8, label=item, mec='k', capsize=2, mfc=colors[d])
    ax3.axvline(lim, linestyle='--', color='grey',linewidth=0.5)
    ax3.axhline(lim, linestyle='--', color='grey',linewidth=0.5)
    ax3.fill_between([0.58,lim],0.58,lim, facecolor='dimgray', alpha=0.05)
    ax3.set_ylim(0.58,1.5)
    ax3.set_xlim(0.58,1.5)
    # ax3.set_title(r'$kT_0 / \bar{kT}$')
    ax3.set_xlabel(r'$RT$')
    ax3.set_ylabel(r'$RT$')

    # col=' n0 (count/arcmin³)'
    # col_er=' n0Err'
    # lim=0.012
    # ax2.errorbar(f_agl[col][d], f_agl[col][d], xerr=f_agl[col_er][d], yerr=f_agl[col_er][d], color=colors[d], marker='.', ms=8, mec='k', capsize=2, mfc=colors[d])
    # ax2.axvline(lim, linestyle='--', color='grey',linewidth=0.5)
    # ax2.axhline(lim, linestyle='--', color='grey',linewidth=0.5)
    # ax2.fill_between([lim,0.038],lim,0.038, facecolor='gray', alpha=0.4)
    # ax2.set_ylim(0,0.038)
    # ax2.set_xlim(0,0.038)
    # ax2.set_title(r'$n_0$')

    legs.append(Line2D([0], [0], marker='.', color='w',markerfacecolor=colors[d], mec='k', markersize=9, alpha=0.7, label=item))

fig.set_size_inches(12.5,3.9)
plt.subplots_adjust(wspace=0.4, hspace=0.4, bottom=.2)
plt.legend(handles=legs,loc='lower right',bbox_to_anchor=(0.3,-0.3),
       ncol = 6,fancybox=True, shadow=True, fontsize = 'x-small')
# plt.legend(handles=legs,loc='center bottom', bbox_to_anchor=(1,0.5),
  #       ncol=1, fancybox=True, shadow=True, fontsize = 'x-small')
plt.savefig('CC.png')
plt.close()