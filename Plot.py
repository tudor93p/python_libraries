import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import time, sys, socket
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d,interp2d
from scipy.sparse import csc_matrix,linalg as Sla
import itertools

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavuserif'
#rcParams['font.family'] = 'serif'

import Geometry
import Algebra,Utils,Lattices


color_palette = [turq,pink,blue,red,lblue] = [tuple(np.array(c)/255.) for c in [(126,214,206),(219,190,184),(21,72,112),(163,82,80),(80,126,163)]]

from matplotlib.colors import LinearSegmentedColormap



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

def format_panel_title(panel,bigger_fontsize,title,normal_fontsize=None):
    
    normal_fontsize = Utils.Assign_Value(normal_fontsize, bigger_fontsize)

    if isinstance(panel,int):
        panel = chr(ord('a')+panel)

    part1  = r'{\fontsize{'+str(bigger_fontsize)+r'pt}{3em}\selectfont{}('
    part2 = str(panel) + r')}{\fontsize{'+str(normal_fontsize) 
    part3 = r'pt}{3em}\selectfont{}'+title 

    return part1+part2+part3 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#
    
def reverse_cmap(cmap):

    return cmap[:-2] if (len(cmap)>=2 and cmap[-2:]=="_r") else cmap+"_r" 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



def inset_positions(text=None, width=None, height=None):
   
    width = Utils.Assign_Value(width, 0.3)

    height = Utils.Assign_Value(height, width)

    pad = 0.05

    height = min(height, 1-2*pad)

    width = min(width, 1-2*pad)

    horiz = {"left":pad,"center":1/2-width/2,"right":1-width-pad}


    vert = {"bottom":pad,"middle":1/2-height/2,"upper":1-height-pad}




    if text is None:

        return sorted([" ".join(pair).capitalize() for pair in itertools.product(vert.keys(), horiz.keys())])

    horiz["none"] = horiz["right"]
    vert["none"] = vert["bottom"]

    text1 = text.lower()

    def get_part(D):

        for (k,v) in D.items():

            if k in text1: return v

        return D["none"]

    return [get_part(horiz), get_part(vert), width, height]





def inset_sizes(kwargs, changed_keys):

    def f(k,v):

        if k in changed_keys:

            return v*0.8

        return v

    return {k:f(k,v) for (k,v) in kwargs.items()}
    




def inset_fontsize(rectangle=None, fontsize=10):
    
#    fontsize1 = fontsize*np.sqrt(max(width,height))
    return fontsize * 0.8

def add_inset_axes(ax, rectangle, axisbg='w', fontsize=10):


    fig = plt.gcf()

    box = ax.get_position()

    left, bottom, width, height = rectangle

    inax_position  = ax.transAxes.transform([left,bottom])

    transFigure = fig.transFigure.inverted()

    new_rectangle = np.append(transFigure.transform(inax_position),
                                [box.width*width, box.height*height])

    subax = fig.add_axes(new_rectangle, facecolor=axisbg)


    fontsize1 = inset_fontsize(rectangle, 10 if fontsize is None else fontsize)

 

#    for (lab, size, axis) in [(subax.get_xticklabels, width, subax.xaxis),
#                              (subax.get_yticklabels, height, subax.yaxis)]:

#        labelsize = lab()[0].get_size() * size**0.5

#        axis.set_tick_params(labelsize=labelsize)

    for axis in [subax.xaxis, subax.yaxis]:

        axis.label.set_size(fontsize1)
       
    subax.tick_params(labelsize=fontsize1)
    



    return subax


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

def disable_labels(ax):

    if ax is None:

        return 

    if np.ndim(ax)>0:

        return [disable_labels(a) for a in ax]


    ax.set(xticklabels=[],yticklabels=[],xticks=[],yticks=[],
            title="",xlabel=None,ylabel=None)

    legend = ax.get_legend()
    
    if legend is not None:
        
        legend.remove()

    return 





def set_fontsize(ax, fontsize):

    if fontsize is None:
        return 

    if np.ndim(ax)>0:

        return [set_fontsize(a, fontsize) for a in ax]


    ax.title.set_size(fontsize)

    ax.tick_params(labelsize=fontsize)

    ax.xaxis.label.set_size(fontsize)

    ax.yaxis.label.set_size(fontsize) 

    return


#===========================================================================#
#
# Collect legends
#
#---------------------------------------------------------------------------#

def collect_legends(*axes):

    lines, labels = axes[0].get_legend_handles_labels()

    for ax in axes[1:]:

        li, la = ax.get_legend_handles_labels()

        lines = lines + li
        labels = labels + la


    return lines,labels





#===========================================================================#
#
# colormap transparency
#
#---------------------------------------------------------------------------#


def fadeaway_cmap(cmap,minalpha=0,maxalpha=1):

    ncolors = 256
    color_array = plt.get_cmap(cmap)(range(ncolors))

    color_array[:,-1] = np.linspace(minalpha,maxalpha,ncolors)**1.5

    return LinearSegmentedColormap.from_list(name='aux',colors=color_array)


def prep_ticklab(y, nmax_digits=2):

    if nmax_digits==0:

        return str(int(y))


    if Utils.NrDigits(y)==0: 
       
        s = str(y)
        
        i = s.find(".")

        return s if i==-1 else s[:i]




    s = str(round(y,nmax_digits)).rstrip("0").rstrip(".")
    
    i = s.find(".")+1
   
    return s[:i+nmax_digits] if i>0 else s 


def prep_ticklabs(ticks, nmax_digits=2):

    labels = [prep_ticklab(ticks[0], nmax_digits)]

    for i in range(1,len(ticks)):

        labels.append(prep_ticklab(ticks[i], nmax_digits))

        if nmax_digits<8 and labels[i]==labels[i-1]:

            return prep_ticklabs(ticks, nmax_digits+1)

    return labels 


#===========================================================================#
#
# colorbar with nice ticks
#
#---------------------------------------------------------------------------#

def good_colorbar(plot,vminmax,ax,label=None,digits=2,
        ticks=None,**kwargs):

  vmin,vmax = Utils.Round_toMaxOrder(vminmax,digits)

  if vmin==vmax:
    return 


  T = np.unique([vmin,vmax] if vmin*vmax >= 0 else [vmin,0,vmax])

  cbarticks = T 

  if ticks is not None:
#      cbarticks = np.append(cbarticks,np.setdiff1d(ticks,cbarticks))
    ticks_ = np.hstack(ticks) 
    ticks_ = ticks_[np.logical_and(ticks_>=vminmax[0],ticks_<=vminmax[1])]

    cbarticks = np.unique(np.append(cbarticks,ticks_)) 





  cbar = ax.get_figure().colorbar(plot,ax=ax,
          boundaries=np.linspace(vmin,vmax,100),
          ticks=cbarticks, 
          drawedges=False)


  n = max(Utils.NrDigits(vmin),Utils.NrDigits(vmax))
    
  cbarticklabels = prep_ticklabs(cbarticks, n)
  

  cbar.ax.set_yticklabels(cbarticklabels,**kwargs)

  if label is not None and len(label):

    cbar.set_label(label, rotation=90, **kwargs)









#===========================================================================#
#
# Helper function to plot against the Ox or Oy axis
#
#---------------------------------------------------------------------------#


  


def get_plotxy(axis=0):


  if axis not in [0,1]:
    raise ValueError("'axis' should be either 0 or 1")

  def makearray(a):
    try:
      b = np.array(a,dtype=complex)
      return np.array(a)
  
    except:
      return a

  def f(variable,function,Z=None,fill=None):

    if fill is not None:
  
      variable = np.concatenate(([variable[0]],variable,[variable[-1]]))
      function = np.concatenate(([fill],function,[fill]))
 
    xy = [makearray(variable),makearray(function)][::-2*axis+1]


    if Z is not None:

      Z = makearray(Z)
      if axis==1: 
        Z = np.transpose(Z)

      return xy,Z

    return xy
 

  return f


#===========================================================================#
#
# Helper function to extend two limits
#
#---------------------------------------------------------------------------#

def extend_limits(lim,amount=0.04):

  if lim is None:
        
    return [None,None]

  for l in lim:
    if l is None:
      return [None, None]

  return np.array(lim) + np.diff(lim)*np.array([-1,1])*amount






#===========================================================================#
#
# Point s
#
#---------------------------------------------------------------------------#


#1 point == fig.dpi/72.0 pixels
#https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
#https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot-axes-scatter-markersize-by-x-scale/48174228#48174228



#==============================================================================
# update 2D lims according to new point R drawn
#------------------------------------------------------------------------------

def newlims(R,lims):
  lims = [[f(l,r) for l,f in zip(li,[min,max])] for li,r in zip(lims,R)]
  return np.array(lims)

#def distribute_tasks_to_processes(n,p,f):
#
#  operations = [f(i) for i in range(n)]
#
#  nr_oper = sum(operations)
#
#  oper_per_core = int(nr_oper/p)
#
#  def end(start):
#  
#    for end_ in range(start,n):
#
#      if np.sum(operations[start:end_]) >= oper_per_core: return end_
#
#    return n-1
#
# 
#  iis = [[0,end(0)]]
#
#  for _ in range(p-1):
#   
#    aux,start = iis[-1]
#
#    iis.append([start,end(start)])
#
#  return iis
#
#
#
#
#def get_bonds_1(Rs,start,end,bond,tol,out_dict):
#
#  bonds = []
#
#  for i in range(start,end):
#
#    rest_atoms = Rs[i:]
#
#    distances = np.abs(la.norm(rest_atoms-Rs[i],axis=1)-bond) 
#
#    for r2 in rest_atoms[np.nonzero(distances < tol)]:
#
#      bonds.append(np.vstack((Rs[i],r2)).T)
#
#  out_dict[start] = bonds
#
#from JobsClass import Jobs
#from multiprocessing import Process,Manager
#
#def get_bonds(Rs,bond,processes=8,tol=1e-4):
#
#  n  = len(Rs)
#
#  out_dict =  Manager().dict()
#
#  jobs = Jobs(get_bonds_1,processes,Progress_Manager=0,Pre_Args=[Rs],Post_Args=[bond,tol,out_dict])
#
#  limits = distribute_tasks_to_processes(n,processes,lambda i:max(0,n-i-1))
#
#  for start,end in limits:
#
#    jobs.Add(start,end)
#
#
#  jobs.Execute()
#
#
#  return [item for q in out_dict.values() for item in q]



def get_bonds(allRs, bond, dim=None, tol=1e-5):


  dim = Utils.Assign_Value(dim,len(allRs[0]))


  def bonds(Rs,bond=bond,dim=dim,tol=tol):

    N = 6000    # for memory reasons

    n = int(np.ceil(len(Rs)/N))

    N = int(np.floor(len(Rs)/n))

    def get_pairs(b,c):
    
        i,j = [np.arange(k*N,min((k+1)*N,len(Rs))) for k in (b,c)]

        d = np.isclose(Algebra.OuterDist(Rs[i,:],Rs[j,:]),bond,atol=tol)
   
        f = lambda x: x if b!=c else np.triu(x,1)

        return np.argwhere(f(d)) + np.array([b,c])*N



    pairs = np.vstack([get_pairs(b,c) for b in range(n) for c in range(b+1)])
        
    return np.transpose(Rs[pairs,:dim],axes=(0,2,1)).reshape(-1,dim)



  if dim == len(allRs[0]):

    return [bonds(allRs)]

  
  if dim == len(allRs[0])-1:

    layers = np.unique(allRs[:,-1].round(int(-np.log10(tol))+1))

    return [bonds(allRs[np.abs(allRs[:,-1]-L)<tol]) for L in layers]







#===========================================================================#
#
# Infer whether we should add to ax, show/save a figure
#
#---------------------------------------------------------------------------#
 
def get_methodaxfname(ax_fname): 

  if type(ax_fname) == str:
    method = "save"; fname = ax_fname

  elif type(ax_fname)==type(None):
    method = "show"; fname = None

  else:
    method = "add";   ax = ax_fname; fname = None


  if method in ["save","show"]:

    fig = plt.figure(figsize=(3+3/8,(3+3/8)))

    ax = fig.add_subplot(1,1,1)


  return method,ax,fname
 
  
#===========================================================================#
#
# Infer how many unit cells should be plotted
#
#---------------------------------------------------------------------------#



def get_plot_cells(f,ns_or_UCs):

  typ_inp, *inp = Utils.Assign_Value(ns_or_UCs,["ns",1])

  if typ_inp == "ns":  return f(*inp)

  if typ_inp == "UCs": return inp[0]

  raise ValueError("Please provide [type,input] for the third argument 'ns_or_UCs', where type is either of 'ns' or 'UCs'.")




def get_plot_UCs(Latt,ns_or_UCs=None):

  return get_plot_cells(Latt.get_UCs,ns_or_UCs)


def get_plot_rUCs(Latt,ns_or_UCs=None):

  return get_plot_cells(Latt.get_rUCs,ns_or_UCs)




#===========================================================================#
#
# LDOS 
#
#---------------------------------------------------------------------------#

def LDOS(plot_data,ax_fname=None,Latt=None,ns_or_UCs=None,cbarlabs=None,plotmethod="pcolor",vminmax=None,axtitle="",nr_pcolor=250,cmaps=["viridis"],dotsize=60,zorder=0,fontsize=20,axwidth=1.2,dpi=300, show_colorbar=True):

#):#,bondcols=color_palette,atomsize=50,atomsymb=None,sublattcols=['k','r','b','y'],):


  method,ax,fname = get_methodaxfname(ax_fname) 

  if type(Latt) == type(None):
#    Rs = np.zeros((len(plot_data),1,2),dtype=float)
    Rs = np.zeros((1,2),dtype=float)
  else:
    Rs = get_plot_UCs(Latt,ns_or_UCs)


  Rs = [Rs  for p in plot_data]

  cbarlabs = Utils.Assign_Value(cbarlabs,["" for p in plot_data])

  def xylim(Rs_):
    def xyminmax(xyc,r):

      x,y = (xyc[:,:2]+r).T

      return [np.min(x),np.max(x),np.min(y),np.max(y)]

    mM = [xyminmax(xyc,r) for (xyc,R) in zip(plot_data,Rs_) for r in R ]

    mM = [f(x) for (f,x) in zip([min,max,min,max],np.vstack((mM)).T)]

    return np.split(np.array(mM),2)

  
  def newC(xyc,R,newXY=None):

    R = np.vstack((R.T,np.zeros(len(R))))

    XYC = [Algebra.FlatOuterSum(*a) for a in zip(xyc.T,R)]

    if type(newXY) == type(None):
      return XYC

    return  interp2d(*XYC,kind='cubic')(*newXY)

  vmin,vmax = Utils.Round_toMaxOrder(Utils.Assign_Value(vminmax,(0, max([max(xyc[:,-1]) for xyc in plot_data]))),2)


  if plotmethod == "pcolor":

    l = np.array([np.diff(mM)[0] for mM in xylim()])
  
    ns = np.maximum(3, np.array(l/max(l)*nr_pcolor, dtype=int))
  
    newXY = [np.linspace(*mM[::-1],n) for (mM,n) in zip(xylim(Rs),ns)]

    newCs = [newC(*it,newXY) for it in zip(plot_data,Rs)]

    newXYs = [np.meshgrid(*newXY) for item in plot_data]

  elif plotmethod == "scatter":

    aux = [newC(*it) for it in zip(plot_data,Rs)]

    newCs = [C for (X,Y,C) in aux]

    newXYs = [[X,Y] for (X,Y,C) in aux] 
    


  for iC,(newXY,C,cmap,title) in enumerate(zip(newXYs,newCs,cmaps,cbarlabs)):


    if len(newCs) > 1:
      ncolors = 256
      color_array = plt.get_cmap(cmap)(range(ncolors))
      color_array[:,-1] = np.linspace(0.0,1.0,ncolors)**1.5
      cmap = LinearSegmentedColormap.from_list(name='aux',colors=color_array)


    if plotmethod == "pcolor":
#      C = np.ma.array(C,mask=(iC != np.argmax(newCs,axis=0)))

      plot = ax.pcolormesh(*newXY,C,vmin=vmin,vmax=vmax,cmap=cmap,edgecolors='face',shading='flat')
      
   
    elif plotmethod == "scatter":

      plot = ax.scatter(*newXY ,cmap=cmap, s=dotsize, c=C,zorder=zorder, linewidth=0, vmin=vmin,vmax=vmax)

    if show_colorbar:

        cbar = good_colorbar(plot, [vmin,vmax], ax, label = title,
                    fontsize = fontsize,
                )


  if len(axtitle)>0: ax.set_title(axtitle,fontsize=fontsize)


  xyl = xl, yl =  [ax.get_xlim(), ax.get_ylim()]

  D = [np.diff(l)[0] for l in xyl]

  i, I = np.argsort(D)

  delta, Delta = D[i], D[I]/3 # axes ratio shouldn't be worse than this nr

  if delta < Delta:

    [ax.set_xlim, ax.set_ylim][i](xyl[i]/(1e-8+delta) * Delta)

  ax.set_aspect(1)

  ax.set_xlabel("$x$",fontsize=fontsize)
  ax.set_ylabel("$y$",rotation=0,fontsize=fontsize)

  ax.tick_params(width=axwidth, length= 3,labelsize=fontsize*0.85)


  [ax.spines[S].set_linewidth(axwidth) for S in ['top','bottom','left','right']]
  if method == "save":
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    plt.savefig(fname+".png",dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()

#===========================================================================#
#
#
#---------------------------------------------------------------------------#

def Lattice(Latt, ax_fname=None, ns_or_UCs=None, sublattices=None, 
        mesh=False, bondlength=None, bondwidth=2, bondcols=color_palette, 
        atomsize=50, atomsymb=None, sublattcols=['k','r','b','y'], zorder=0,
        fontsize=20, axwidth=1.2, dpi=300, labels=False, alpha_atom=0.5, 
        atom_limit=5000):


  method,ax,fname = get_methodaxfname(ax_fname) 

  UCs = get_plot_UCs(Latt,ns_or_UCs)


  plotted_sublattices = Latt.Sublattices_contain(sublattices)


  nr_atoms_UC = len(Latt.PosAtoms(plotted_sublattices)) 
  nr_UCs = len(UCs)
  nr_atoms = nr_atoms_UC*nr_UCs


  
#  ax.set_title(str(nr_atoms_UC)+" atoms/UC * "+str(nr_UCs)+" UCs = "+str(nr_atoms),fontsize=fontsize)

  z=0

  if (atom_limit is None) or (nr_atoms < atom_limit):

 

  # -------------------- plot bonds under atoms --------------------------- #




    if bondwidth != 0: 
  
      bondlength = Utils.Assign_Value(bondlength,lambda:Latt.Distances()[1],[])
  
 
      atoms = Algebra.FlatOuterSum(Latt.PosAtoms(sublattices),UCs)

  
      all_bonds = get_bonds(atoms,bondlength,2)
  
  
      for z,(bonds,bondcol) in enumerate(zip(all_bonds,bondcols)):
  
  		#this takes a lot of time
        ax.plot(*bonds,color=bondcol,lw=bondwidth,zorder=zorder+z) 
  
  
  
  
  
  
    # ------------------------- plot atoms ---------------------------------- #
  

    sublattcols = np.repeat(np.reshape(sublattcols,(1,-1)),max(1,int(np.ceil(len(Latt.Sublattices.keys())/len(sublattcols)))),axis=0).reshape(-1)


    for sublatt,atcol in zip(sorted(list(Latt.Sublattices.keys())),sublattcols):
  
      if sublatt in plotted_sublattices:
  
        atoms = Latt.PosAtoms(sublatt)
  
        if len(atoms):
  
          for zi,(ucs,size,alpha) in enumerate(zip([UCs,UCs[Algebra.FlatOuterDist(UCs)<1e-5,:]],[0.85,1.05],[alpha_atom,1])):
  
            X,Y,*rest = Algebra.FlatOuterSum(ucs,atoms).T
  
            ax.scatter(X,Y,s=atomsize*size**2,marker=atomsymb,color=atcol,linewidths=0,edgecolor='none',zorder=zorder+z+1+zi,alpha=alpha,label=sublatt if zi==1 else None)
            break

#            print(Algebra.minmax(X),Algebra.minmax(Y))

  # ---------------- plot unit cells below lattice ----------------------- #

  else:
    mesh = True
 
  if Latt.LattDim == 2 and mesh==True:

    a1,a2 = Latt.LattVect[:2,:2]
    for a in [a1,a2]:
      ax.arrow(0,0,a[0],a[1],head_width=.11,length_includes_head=True,color='k',zorder=zorder+z+2)


    add = Geometry.Order_PolygonVertices([[0,0],a1,a2,a1+a2])

    xlim,ylim = ax.get_xlim(), ax.get_ylim()


    UCs = Algebra.FlatOuterSum(Latt.get_UCs(1),UCs)


    for i in reversed(range(len(UCs))):

      
      if sum(la.norm(UCs - UCs[i],axis=1)<1e-6) >1 :

        UCs = np.delete(UCs,i,axis=0)

    for uc in UCs[:,:2]:

      ax.fill(*(uc+add).T,color='k',zorder=zorder-1,alpha=0.02+0.15*np.random.rand(),ec=None,lw=0)

    ax.set_xlim(xlim); ax.set_ylim(ylim) 

  if labels and len(plotted_sublattices)<5:

    ax.legend(loc='center',fontsize=fontsize)


  xyl = xl, yl =  [ax.get_xlim(), ax.get_ylim()]

  D = [np.diff(l)[0] for l in xyl]

  i, I = np.argsort(D)

  delta, Delta = D[i], D[I]/3 # axes ratio shouldn't be worse than this nr

  if delta < Delta:

    [ax.set_xlim, ax.set_ylim][i](xyl[i]/(1e-8+delta) * Delta)

  ax.set_aspect(1)
 
  ax.set_aspect(1)

  ax.tick_params(width=axwidth, length= 3,labelsize=fontsize*0.85)

  [ax.spines[S].set_linewidth(axwidth) for S in ['top','bottom','left','right']]
 

  
  if method == "save":
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    plt.savefig(fname+".png",dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()




#===========================================================================#
#
#
#---------------------------------------------------------------------------#

def BrillouinZone(Latt,ax_fname=None,ns_or_UCs=None,tol=1e-5,bondwidth=2,bondcol=lblue,dotcol=red,dotsize=50,fill=0.0,show_vectors=False,zorder=0,fontsize=20,axwidth=1.2,dpi=300):



  
  method,ax,fname = get_methodaxfname(ax_fname) 

  UCs = get_plot_rUCs(Latt,ns_or_UCs)



  xylims = [ax.get_xlim(),ax.get_ylim()]





  # ---------------- determine the Brillouin zone nodes ------------------- #


  BZ = Latt.BrillouinZone()

  BZc = np.append(BZ,BZ[:1],axis=0)


  Atoms = Algebra.FlatOuterSum(BZ,UCs)
#  Atoms=np.unique(Algebra.FlatOuterSum(BZ,UCs).round(int(np.log10(1/tol))),axis=0)

#  if np.count_nonzero(Algebra.OuterDist(Atoms,Atoms) < tol) != len(Atoms):
#    print("\n*** Warning: there are duplicates in the node list ***\n")



#  ax.plot(*np.append(BZ,BZ[:1],axis=0).T,color='k',alpha=0.15)#,ec=None,lw=0)



  # -------------------- plot bonds under atoms --------------------------- #
 
  if bondwidth != 0.0 and len(Atoms) < 300:
    for bondlength in set(la.norm(np.diff(BZc,axis=0),axis=1).round(int(np.log10(1/tol)))):

      for bonds in get_bonds(Atoms,bondlength,2):

        ax.plot(*bonds,color=bondcol,lw=bondwidth,zorder=zorder) 




  # ------------------------- plot atoms ---------------------------------- #

  if len(Atoms)< 2000:
    X,Y,*rest = Atoms.T
  
    ax.scatter(X,Y,s=dotsize,color=dotcol,linewidths=0,edgecolor='none',zorder=zorder+1)
  
    for a in Atoms:
      xylims = newlims(a,xylims)

#  # ---------------- plot unit cells below lattice ----------------------- #

 

  if show_vectors == True:    
    for a in Latt.ReciprocalVectors()[:2,:2]:
      ax.arrow(0,0,a[0],a[1],head_width=.05*np.sqrt(dotsize),length_includes_head=True,color='k',zorder=zorder+2)
  
      xylims = newlims(a,xylims)

  ax.fill(*BZc.T,color='k',zorder=zorder-1,alpha=fill,ec=None,lw=0)

    
  ax.set_aspect(1)
  ax.tick_params(width=axwidth, length= 3,labelsize=fontsize*0.85)
  [ax.spines[S].set_linewidth(axwidth) for S in ['top','bottom','left','right']]

  xlim,ylim = [l+ abs(np.subtract(*l))*np.array([-1,1])*0.1 for l in xylims]

  ax.set_xlim(xlim); ax.set_ylim(ylim) 

 
  if method == "save":
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    plt.savefig(fname+".png",dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()

#===========================================================================#
#
# Compare two Brillouin zones
#
#---------------------------------------------------------------------------#

def Compare_BrillouinZone(Latt,ax_fname=None,linewidth=1,col=lblue,dotsize=None,zorder=0,fontsize=8,axwidth=1.2,dpi=300):

 
 
  method,ax,fname = get_methodaxfname(ax_fname) 

 
  dotsize = Utils.Assign_Value(dotsize,linewidth**2)

  

  rUCs = Latt.get_rUCs(0)
  z = 0
 
  Cs,Ns = Latt.Components

  if len(Cs) and len(Ns):
  
    rUCs = Latt.get_rUCs(int(np.ceil(np.amax(Ns)/2)))
  
    
  
    def inside(bz,BZ):
      for shift in 1e-4*Utils.vectors_of_integers(Latt.LattDim,1):
        c = 0 
        for r in bz+shift:
          if Geometry.PointInPolygon_wn(r,BZ) != 0:
            c = c + 1
            if c == 2:
              return 1
      return 0
  
  
    smallBZ = Latt.BrillouinZone()
  
    which = []
  
    for (z,C) in enumerate(Cs):
  
      C.Plot_BrillouinZone(ax_fname=ax,ns_or_UCs=["ns",0],fontsize=fontsize,zorder=zorder+z*3,dotsize=0,bondwidth=0,bondcol="gray",fill=0.15)
  
      largeBZ = C.BrillouinZone()
  
      which.append([inside(uc+smallBZ,largeBZ) for uc in rUCs[:,:Latt.LattDim]])
       
  
    rUCs = rUCs[np.nonzero(np.sum(which,axis=0))]



  Latt.Plot_BrillouinZone(ax_fname = ax,ns_or_UCs=["UCs",rUCs],fontsize=fontsize,zorder=zorder+(z+1)*3,dotsize=dotsize,dotcol=col,bondcol=col,fill=0,bondwidth=linewidth,show_vectors=True)





  ax.set_aspect(1)
  ax.tick_params(width=axwidth, length= 3,labelsize=fontsize*0.85)
  [ax.spines[S].set_linewidth(axwidth) for S in ['top','bottom','left','right']]











  if method == "save":
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    plt.savefig(fname+".png",dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()




#===========================================================================#
#
#
#---------------------------------------------------------------------------#


def DOScolor(data,ax_fname=None):

  method,ax,fname = get_methodaxfname(ax_fname) 











  if method == "save":
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    plt.savefig(fname+".png",dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()



#===========================================================================#
#
# Plot a line/ scatter points
#
#---------------------------------------------------------------------------#


def Line(points,ax_fname=None,dpi=300,**kwargs):

  method,ax,fname = get_methodaxfname(ax_fname) 

  ax.plot(*np.array(points).T[0:2],**kwargs)


  if method == "save":
    plt.tight_layout()
    plt.savefig(fname,dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()




def Points(points,ax_fname=None,dpi=300,ylim=None,**kwargs):

  method,ax,fname = get_methodaxfname(ax_fname) 

  ax.scatter(*np.array(points).T[0:2],linewidths=0,**kwargs)

  ax.set_ylim(Utils.Assign_Value(ylim,ax.get_ylim()))

  if method == "save":
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    plt.savefig(fname+".png",dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()

  plt.close()

def Text(ax,coord,text,**kwargs):

  if np.ndim(coord) ==1: coord = [coord]

  for (x,y,*rest) in coord:
  
    ax.text(x,y,text,**kwargs)

#===========================================================================#
#
# Plot all combinations of the sublattices and layers
#
#---------------------------------------------------------------------------#


#  l1,l2 = [list(itertools.combinations(Latt.Sublattices.keys(),n)) for n in [2,4]]
#
#  l = l1+l2
#
#  nrows = max(int(np.ceil(len(l)/4)),1)
#  ncols = max(int(np.ceil(len(l)/nrows)),1)

#===========================================================================#
#
# Plot all combinations of the sublattices and layers
#
#---------------------------------------------------------------------------#

def Lattice_and_Vacancies(Latt,ax_fname=None,ns_or_UCs=None,atomsize=10,bondwidth=None,mesh=False,atomsymb=None,vacancysymb='X',zorder=0,dpi=300,vacancyfactor=0.0,**kwargs):


  method,ax,fname = get_methodaxfname(ax_fname) 

  ns_or_UCs = ["UCs", get_plot_UCs(Latt,ns_or_UCs)]



  Lattice(Latt,ax_fname=ax, ns_or_UCs=ns_or_UCs, atomsize=atomsize, atomsymb=atomsymb,bondwidth=bondwidth,mesh=mesh,zorder=zorder,**kwargs)

  if vacancyfactor > 0 and len(Latt.VacancyLatt().PosAtoms()) > 0:
  
    Lattice(Latt.VacancyLatt(),ax_fname=ax, ns_or_UCs=ns_or_UCs, atomsize=atomsize*vacancyfactor**2,atomsymb=vacancysymb,bondwidth=0,mesh=False,zorder=zorder+1,**kwargs)
  

 


  if method == "save":
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    plt.savefig(fname+".png",dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()


#===========================================================================#
#
# Plot DOS
#
#---------------------------------------------------------------------------#

def DOS(plot_data, ax_fname=None,ylim=1,xlim=None,fontsize=20,color=blue,cmap="viridis",savgol=None,NrEn=1000,cbarlim=None,linewidth=2,dotsize=6,zorder=3,axwidth=1.5):

#, xticks=None, xticklabels=None,ylim=1,fontsize=20,color=blue,dotsize=1,zorder=3,cmap="viridis",cbarlabel="",cbarlim=0):

  method,ax,fname = get_methodaxfname(ax_fname) 

  ax.set_title("DOS",fontsize=fontsize)

  def newfun(oldx,fun,newx):
   
    if len(fun) == 0:
      return fun
 
    if type(savgol) == tuple:
      fun = savgol_filter(fun,*savgol)

    return interp1d(oldx,fun)(newx)

  def process_data():
    y0, *rest = plot_data

    y = np.linspace(np.amin(y0),np.amax(y0),NrEn)

    rest = [newfun(y0,f,y) for f in rest]
 
    concat = lambda q: np.concatenate(q) if len(q) else []

    return np.tile(y,len(rest[0::2])),concat(rest[0::2]),concat(rest[1::2])


  plot_y, plot_x, plot_c = process_data()



  if len(plot_c) == 0:    
    ax.plot(plot_x,plot_y,c=color,lw=linewidth,zorder=zorder)    

  else:


    p,q = np.array(cbarlim)/np.diff(cbarlim)[0]
#    a(1+p) == p b, qa  == b(q-1)

    if abs(q) > abs(p):
      vmin,vmax = np.array([p/q, 1])*max(plot_c)
   
    else:
      vmin,vmax = np.array([1, q/p])*min(plot_c)


    ax.scatter(plot_x, plot_y, c=plot_c, cmap=cmap, s=dotsize, zorder=zorder, linewidth=0,vmin=vmin,vmax=vmax)

#    colorbar = plt.gcf().colorbar(scatter, ax=ax)#, ticks=cbarticks,boundaries=np.linspace(min(cbarticks),max(cbarticks),100),fraction=0.045)



#  ax.set_ylabel("Energy",rotation=90,fontsize=fontsize)


  ax.tick_params(width = axwidth, length= 6,labelsize=fontsize*0.85)
  [ax.spines[S].set_linewidth(axwidth) for S in ["top","bottom","left","right"]]


#  xticks = Utils.Assign_Value(xticks,np.linspace(min(plot_x),max(plot_x),5))

#  ax.set_xticks(xticks)

#  ax.grid(True,axis="x",linestyle="--",zorder=zorder-1)

#  ax.set_xticklabels(Utils.Assign_Value(xticklabels,np.round(xticks,2)))


  if type(ylim) in [int,float]:

    m,M = np.amin(plot_y),np.amax(plot_y)

    ylim = (M+m)/2 + np.array([-1/2,1/2])*(M-m)/ylim

  ax.set_ylim(ylim)




  whichy = np.array([ylim[0] <= y <= ylim[1] for y in plot_y])

  if sum(whichy) ==0:
    whichy = np.arange(len(plot_y),dtype=int)

  
  ax.set_xlim(0,Utils.Assign_Value(xlim,np.amax(np.array(plot_x)[whichy])*1.1))


  ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='x')

  
  
  if method == "save":
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    plt.savefig(fname+".png",dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()
  



#===========================================================================#
#
# Plot 2D spectrum on a path
#
#---------------------------------------------------------------------------#

def Spectrum_2D_Path(plot_data, ax_fname=None, xticks=None, xticklabels=None,ylim=1,fontsize=20,color=blue,dotsize=1,zorder=3,cmap="viridis",cbarlabel="",cbarlim=None):


  method,ax,fname = get_methodaxfname(ax_fname) 



  plot_x, plot_y, plot_c = plot_data




  if len(plot_c) == 0:
    
    ax.scatter(plot_x, plot_y, c=color, s=dotsize, zorder=zorder, linewidth=0)

  else:

    cbarticks = Utils.Assign_Value(cbarlim,[min(plot_c),max(plot_c)])

    if max(cbarticks)-min(cbarticks) < 1e-4:
   
      cbarticks = [-1.01,1.01]

    cbarticks = np.append(cbarticks,0)

    cbarticks = np.unique(np.round(cbarticks,2))

    scatter = ax.scatter(plot_x, plot_y, cmap=cmap, c=plot_c, s=dotsize, zorder=zorder, linewidth=0, vmin=min(cbarticks), vmax=max(cbarticks))

   



    colorbar = plt.gcf().colorbar(scatter, ax=ax, ticks=cbarticks,boundaries=np.linspace(min(cbarticks),max(cbarticks),100),fraction=0.045)

    colorbar.ax.set_yticklabels(np.round(cbarticks,1),fontsize=fontsize)
#
    if len(cbarlabel):
##      colorbar.set_label(cbarlabel,fontsize= fontsize,rotation=0)
      ax.set_title("Colorbar: "+cbarlabel,fontsize=fontsize)

 

  ax.tick_params(width = 1.5, length= 6,labelsize=fontsize*0.85)


#  ax.set_ylabel("Energy",rotation=90,fontsize=fontsize)


  [ax.spines[S].set_linewidth(1.5) for S in ["top","bottom","left","right"]]


  xticks = Utils.Assign_Value(xticks,np.linspace(min(plot_x),max(plot_x),5))

  ax.set_xticks(xticks)


  ax.grid(True,axis="x",linestyle="--",zorder=zorder-1)



  ax.set_xticklabels(Utils.Assign_Value(xticklabels,np.round(xticks,2)))


  if type(ylim) in [int,float]:

    m,M = np.amin(plot_y),np.amax(plot_y)

    ylim = (M+m)/2 + np.array([-1/2,1/2])*(M-m)/ylim

  ax.set_ylim(ylim)


  ax.set_xlim(min(xticks),max(xticks))

 
  
  
  if method == "save":
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    plt.savefig(fname+".png",dpi=dpi)

  elif method == "show":
    plt.tight_layout()
    plt.show()
  

#==============================================================================
# Plot 1D spectrum
#------------------------------------------------------------------------------


def Spectrum_1D(H,k_Point_List,Nr_kPoints):


  K_Points,ticks = path_connect(k_Point_List,Nr_kPoints)

  Ens = np.array([la.eigvalsh(H(k)) for k in K_Points]).T
  
  x = K_Points
  
  col1 = blue

  for y in Ens:
    plt.scatter(x,y,color=col1,s=1)
  
  plt.show()



#==============================================================================
# Plot 0D spectrum
#------------------------------------------------------------------------------



def Spectrum_0D(H,filename,shown_ens = 15,fontsize = 16,size = 20,suptitle=None):

  fig = plt.figure('filename',figsize=(3+3/8,1.6))

  ax = fig.add_subplot(111)

  if suptitle != None:  plt.suptitle(suptitle,fontsize=fontsize)

  [ax.spines[S].set_linewidth(1.2) for S in ['top','bottom','left','right']]


  ax.tick_params(width = 1.2, length= 3,labelsize=fontsize*0.85)


  y = la.eigvalsh(H)


  mid = len(y)//2

  y = y[mid-shown_ens:mid+shown_ens]

  
  x = np.arange(len(y))+mid-shown_ens

  marker_size = [size*(1+ 0.5*(i in [mid-1,mid])) for i in x]

  
  col1 = red

  ax.scatter(x,y,color=col1,s=marker_size,zorder=2)
  
  ax.plot(x,x*0.,'k--',lw=1,zorder=1,alpha=0.4)

  ax.set_xlabel('Eigenvalue number $n$',fontsize=fontsize)
  ax.set_ylabel('$E_n$',fontsize=fontsize*1.1,rotation=0)
 
#  plt.tight_layout()
 
  plt.subplots_adjust(left=0.18,bottom=0.27)
   
  plt.savefig(filename)




#===========================================================================#
#
#  Plot probability density on 2D lattice
#
#---------------------------------------------------------------------------#

def Prob_Dens_2D(XYZs,filename,same_plot=False,suptitle=None,titles=None,exp_color=.37,size=38,fontsize=20,ZmM=[0,1],cmaps = ['Reds','Blues','Greens','Purples','YlOrRd','YlGnBu']):


  padding = 0.125




  color_intens = lambda Z: (Z**exp_color)*0.98+0.02



  if titles != None: titles = [t.replace('\ket','') for t in titles]
  if suptitle != None: subtitle = suptitle.replace('\ket','') 


  XYZs = np.array(XYZs)

  if len(XYZs.shape) == 2: XYZs = np.array([XYZs])

  nr_plots = len(XYZs) if same_plot == False else 1

  def lim(s):
    slim  = m,M = np.array([np.amin(s),np.amax(s)])
    return slim + np.array([-1,1])*(M-m)*padding

  Xlim = lim([X for X,Y,Z in XYZs])
  Ylim = lim([Y for X,Y,Z in XYZs])


  Zmin,Zmax =ZmM



  def marker_size(Z):

    sizes = size*(Z+1)

    if not same_plot: return sizes

    return sizes * (np.amax(XYZs[:,2,:],axis=0) == Z)



  fig = plt.figure(str(np.random.randint(0,10000)),frameon=False,figsize=(3+3/8,(3+3/8)/nr_plots+0.18))

  if suptitle != None: plt.suptitle(suptitle)

  for i_plot in range(len(XYZs)):
   
    i_ax = 0 if same_plot else i_plot

    ax = fig.add_subplot(1,nr_plots,i_ax+1)

    if titles != None: ax.set_title(titles[i_ax],fontsize=fontsize)

    X,Y,Z = XYZs[i_plot]
    cmap = cmaps[i_plot]
  
    [ax.spines[S].set_linewidth(1.2) for S in ['top','bottom','left','right']]
  
  
   
    ax.set_xlim(Xlim)
    ax.set_ylim(Ylim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_aspect(1)
    
  
  
  #  plot21 = Ax2.scatter(X,Y,s= size ,c= np.ones(X.shape)*.3,cmap='Greys',vmin=Zmin,vmax=Zmax,linewidths =0.,edgecolors = 'k',zorder=1,alpha=1) 
  
  
   
  
  #  plot22 = [
    ax.scatter(X,Y,s=marker_size(Z),c=color_intens(Z),cmap=cmap,vmin=Zmin,vmax=Zmax,linewidths =0,edgecolors = 'k',zorder=2,alpha=.9)
  # for Z,cm in zip(Zs,cmaps)]
  
  
#  plt.tight_layout() 
  plt.subplots_adjust(left=0.01,right=.99,bottom=0.0)

  plt.savefig(filename)

















#############################################################################









