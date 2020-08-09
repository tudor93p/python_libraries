import numpy as np
import PlotPyQt



def funfig(obj,Fig,ax): 


  xs = np.linspace(0.,10.,300) # xgrid


  # get the value of the slider
  ys = np.cos(xs*obj.get_slider("k") + obj.get_slider("phi")) 

  ys = ys + float(obj.get_text("dy")) # shift the values



  ax.plot(xs,ys,c=obj.get_combobox("c")) # plot data

  ax.set_ylim([-2,2])












fig = PlotPyQt.Figure(funfig) # initialize figure instance


ks = np.linspace(1.0,3.0,50) # wavevectors
ps = np.linspace(0.0,2.0,50)*np.pi # phases


fig.add_slider(label="Wavevector",key="k",vs=ks)#,columnSpan=1)

#  main.add_slider(label="Useless slider",key="us",vs=ks,next_row=False)

fig.add_slider(label="Phi",key="phi",vs=ps)

fig.add_text(label="Shift",key="dy",text="0.0",columnSpan=1)

fig.add_combobox(["red","blue","black"],label="Color",key="c",next_row=False)





fig.show()

