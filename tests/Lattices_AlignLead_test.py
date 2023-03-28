#!/home/tudor/apps/anaconda3/bin/python
import PlotPyQt

import numpy as np
import Lattices,Geometry

R = 15


   

def py1(desired_angle, LattAtoms0):

    print("py1")

    angles = np.linspace(0,2*np.pi,200) 

    angle = np.argmin(np.abs(desired_angle-angles))

    center = np.mean(LattAtoms0, axis=0)

    radius_outside = np.max(np.linalg.norm(LattAtoms0-center ,axis=1))*1.2

    circle_outside = np.array([np.cos(angles), np.sin(angles)]).T*radius_outside + center

    lead_pos = circle_outside[angle]

    return circle_outside, lead_pos





def py0(seed):

    print("py0")

    N = np.array([R,R])+2 

    np.random.seed(seed)

    polygon = Geometry.MakePolygonConvex((np.random.rand(10,2)-0.5)*R)

    return N, polygon
 



def py2(N, polygon, desired_angle, setup1=py1):

    print("py2")

    SqLatt = Lattices.Lattice(np.eye(2)).Add_atoms()
    
   # dist_nn = SqLatt.Distances()[1]
    
    
    Latt0 = SqLatt.Shift_Atoms(-(N-1)/2).Supercell(N).Reduce_Dim().Rename_Sublattice("Device")
    
    
    LattAtoms0 = Latt0.PosAtoms()
    
    
    Latt = Latt0.Copy().Remove_atoms(list(filter(lambda a: not Geometry.PointInPolygon_wn(a,polygon), LattAtoms0))).Remove_SingleBonds()
    
    LattAtoms = Latt.PosAtoms()
    
    
    BA = Latt.BoundaryAtoms()
    
#    Latt = Latt.Remove_atoms(BA).Add_atoms(BA,"Boundary") 
    
    
    
    circle_outside, lead_pos = setup1(desired_angle, LattAtoms0)
    
    
    Lead = SqLatt.Copy().Shift_Atoms(lead_pos).Rename_Sublattice("Lead initial position")
    
    LeadAtoms = Lead.PosAtoms()


    return LattAtoms0,LeadAtoms,LattAtoms,Lead,SqLatt,BA


def py3(Lead, sc, LattAtoms, SqLatt):

    print("py3")

    Lead = Lead.Supercell(sc).Reduce_Dim(1)

    LeadAtoms2 = Lead.PosAtoms()

    Lead,Bridge = Lead.Attach_toAtoms(LattAtoms, bonds=np.vstack((SqLatt.LattVect,-SqLatt.LattVect)))
  
    LeadAtoms3 = Lead.PosAtoms()
#    one extra unit cell ?  

    return LeadAtoms2, LeadAtoms3, Bridge 
    

# LeadAtoms2, LeadAtoms3, Bridge  = setup3(Lead, [u,l], LattAtoms, SqLatt) 

def plot0(seed, desired_angle, size_island, ax, setup2, setup3):

    N, polygon = py0(seed) 
    
    LattAtoms0,LeadAtoms,LattAtoms,Lead,SqLatt,BoundaryAtoms = setup2(N, polygon, desired_angle)
    
    
    circle_outside, lead_pos = py1(desired_angle, LattAtoms0)
    
    
    
    ax.plot(*circle_outside.T, c='gray')
    
    
    
    r = np.mean(LeadAtoms, axis=0) - np.mean(LattAtoms,axis=0)
    
    
    r[np.argmin(np.abs(r))]=0
    
    
    u = np.round(r/np.min(np.abs(r[r!=0])))
    
    
    l = np.insert([0],np.argmin(np.abs(u)), size_island+1)
    
    
    
    if setup3 is not None:
    
        LeadAtoms2,LeadAtoms3,Bridge = setup3(Lead, [u,l], LattAtoms, SqLatt)
    
    
        center = np.mean(LattAtoms0, axis=0)
        
        for atom in LeadAtoms2:
        
            q = np.linalg.norm(atom-center)/np.linalg.norm(u)
        
            ax.plot(*np.vstack((atom,atom-q*u)).T,color='gray')
        
        ax.scatter(*LeadAtoms2.T, s=20, zorder=3,c = 'magenta',label="Lead initial")

        print("Lead initial",LeadAtoms2[0])
    
#             Lead.Plot_Lattice(ax_fname=ax,atomsize=20,ns_or_UCs=['ns',0],bondlength=dist_nn,zorder=3,sublattcols=['magenta'])
    
    
    
        ax.scatter(*LeadAtoms3.T, s=40, zorder=4, label="Lead aligned")
   
        print("Lead aligned:",LeadAtoms3[0])

        if Bridge is not None:
            
            ax.scatter(*Bridge.T, s=40, zorder=5, label="Extension",c='r')
            
    
    ax.scatter(*BoundaryAtoms.T, s=70, zorder=7, label="Boundary",c='g')
    
    ax.scatter(*LattAtoms.T, s=70, zorder=6, label="Device",c='gray')
    
#  Latt.Plot_Lattice(ax_fname=ax,bondlength=dist_nn,atomsize=70,sublattcols=['g','gray','r'],zorder=0,bondwidth=0)
#
# 
#
#  def add_first(p):
#  
#    return np.vstack((p,p[0:1,:]))
#  ax.plot(*add_first(Geometry.Order_PolygonVertices(BA)).T)

#  ax.plot(*add_first(polygon).T)

    ax.set_aspect(1)

    ax.legend()
  
  
  


def plot(setup2=py2, setup3=None):


    def funfig(obj,fig,ax): 
    
        plot0(obj.get_slider("seed"), obj.get_slider("angle"), obj.get_slider("m"), ax, setup2, setup3)
  

        
  
#  obj.get_text("")

#  obj.get_combobox("")

#  obj.get_checkbox("")


  
  
    Fig = PlotPyQt.Figure(funfig)#,1,1,tight=True,**kwargs) 
    
    
    
    
    Fig.add_slider(label="angle",key="angle",vs=np.linspace(0,2*np.pi,200))#,columnSpan=1,next_row=True)
    Fig.add_slider(label="m",key="m",vs=range(R+10),v0=2)#,columnSpan=1,next_row=True)
    
    Fig.add_slider(label="seed",key="seed",vs=range(50))
    #Fig.add_text(key=None,label=None,text="",columnSpan=-1,next_row=True,max_width=True,min_width=False,vdiv=False)
    
    #Fig.add_combobox(["red","blue","black"],label="Color",key="c",next_row=False)
    
    
    #Fig.add_checkbox(label=None,key=None,next_row=True,columnSpan=-1,max_width=True,function="passive",vdiv=False,status=False)
    
    
    
    Fig.show()
      
    
    
    

if __name__ == "__main__":

    plot(py2, py3)
    
    


