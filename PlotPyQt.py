#############################################################################
#
#   GUI for plotting any 2D data and manipulates parametes through
#           sliders, drop-down menus, editable text, checkboxes, buttons 
#
#   It also saves screenshots and, 
#            if desired, remembers a past parameter configuration
#
#   Based on the GUI tool developed by Jose Lado,
#
#       project ------  https://github.com/joselado/quantum-honeycomp
#       file    ------  /pysrc/interfacetk/plotpyqt.py
#
#############################################################################


import numpy as np
import sys,os,warnings,datetime
import json
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QGridLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

def resol_str(S):

    Si = np.array(np.round(S), dtype=int)

    px = "x".join(map(str,Si)) 
    
    MP = str(np.round(np.prod(Si)/1e6,1))

    return " = ".join([px+"px", MP+"MP"])


def calc_dpi(S, MP):

    nr_pixels = 1e6*int(MP)   # Megapixels 

    dpi = int(np.round(np.sqrt(nr_pixels/np.prod(S))))

    return dpi, np.array(np.round(S*dpi), dtype=int)


def get_interface(plot_figure,i=0):
    """Return an object that plots stuff"""
    class Window(QDialog):
        def __init__(self, parent=None):
            super(Window, self).__init__(parent)
            self.figure = plt.figure(i)
            # this is the Canvas Widget that displays the `figure`
            # it takes the `figure` instance as a parameter to __init__
            self.canvas = FigureCanvas(self.figure)
#            self.button = QPushButton('Plot')
#            self.button.clicked.connect(self.plot)
            # set the layout
            layout = QGridLayout()

            self.row = 1
            self.column = 0
            self.layout = layout
            self._dynamic_ax = self.canvas.figure.subplots()
            self.layout.addWidget(self.canvas, 1,0,1,0)
            self.setLayout(layout)

            self.max_width=90
            self.min_width=90
            self.slidertext = {}
            self.figure_number = 0 


        def plot(self):
            '''Plot the figure'''

            print("Producing figure...")

            self._dynamic_ax.clear()

            self.figure.clear()

            fig = plot_figure(self) # get that figure

#            plt.tight_layout(h_pad=0.1,w_pad=0.1) # adjust axis


            if fig.number!=self.figure.number: 
                raise ValueError("You must plot in the same figure as the one initialized for the interface, use fig = plt.figure(obj.figure.number) in your function, where obj is the input of your function")

            self._dynamic_ax.figure.canvas.draw()

            self.getwrite_config()

            print("Plotting finished.\n")


        def update_plot(self):

            for name in self.slidertext:
                self.get_slider(name)

            self.getwrite_config()

            if self.get_checkbox("live_update"):
              self.plot()
   




        def Vdiv(self):
            div = QtWidgets.QLabel("")
            div.setStyleSheet("QLabel {background-color: #3e3e3e; padding: 0; margin: 0; border-bottom: 1 solid #666; border-top: 1 solid #2a2a2a;}")
            div.setMaximumWidth(2.5)
        
            return div

        def SliderVal(self,x):

            if not isinstance(x,np.float64):

                return str(x)

            n = 3
            
            if np.abs(x) < 1e-20:
            
                magn = 1
            
            elif 1e-20 < np.abs(x) < 1:
            
                magn = -int(np.ceil(np.abs(np.log10(np.abs(x)))))
            
            else:
                magn = int(np.floor(np.log10(np.abs(x)))) + 1
            
            n -= magn + (magn<0) 
            
            if n <= 0: 
            
                return str(np.array(np.round(x,n),dtype=int))
            
            return str(np.round(x,n))




        def connect_object(self,connect,function):

            if type(function) == str:
              if function == "active":
                function = self.plot

              elif function == "passive":
                function = self.update_plot

              else:
                raise ValueError("'function' should be 'passive', 'active', or a an actual function!")


            if function is not None:
              connect(function)


        def check_key(self,key,label):

            if key is None: 
                if label is None: raise

                return label
       
            return key 

        def jump_on_next_row(self):

            self.row += 1 
            self.column = 0 

        def implement_newrow(self,next_row,vdiv,columnSpan,extra_space=0):

            if next_row==False:
              needed_space = vdiv + max(1,columnSpan) + extra_space 

              if self.column + needed_space > 15:
                next_row=True

            if next_row:
              self.jump_on_next_row()

            else:
                
              if vdiv:
                self.layout.addWidget(self.Vdiv(),self.row,self.column)
                self.column +=1

            
            if columnSpan < 0:
              return 15 - self.column - extra_space 

            return columnSpan 

 
        def add_label(self,label,max_width=True):
          
            if label is not None: 
              lb = QtWidgets.QLabel(label)

              if max_width:
                lb.setMaximumWidth(self.max_width)
  
              self.layout.addWidget(lb,self.row,self.column)

              self.column +=1

              return lb

        def add_widget(self,widget,columnSpan,max_width=True,min_width=False):
 
            self.layout.addWidget(widget,self.row,self.column,1,columnSpan)

            self.column += 1 + (columnSpan>0)*(columnSpan-1)

            if max_width:
              widget.setMaximumWidth(self.max_width)

            if min_width:
              widget.setMinimumWidth(self.min_width)

        # ----------------------------------------- #
        # ------------ configuration -------------- #
        # ----------------------------------------- #

        def config_file_exists(self):

            folder = os.getcwd() + "/pyqt_config/"  

            if os.path.isdir(folder):

                fn = self.get_combobox("nr_config")+".json"

                if os.path.exists(folder+fn):
                    return True

            print("There is no pyqt_config file.")

            return False

        def config_file(self):

            fn = self.get_combobox("nr_config")+".json"

            folder = os.getcwd() + "/pyqt_config/"  

            if os.path.isdir(folder)==False:

              os.mkdir(folder)
            
            return folder + fn 

#            return os.getcwd() + "/pyqt_config.json"

        def must_save_config(self):

            return self.get_combobox("save_config")!="No"

#            return self.get_checkbox("remember_config")



        def read_config(self):

            if self.config_file_exists():
                with open(self.config_file(), "r") as f:

                    return json.load(f)
#                    try: return json.load(f)

#                    except: pass

            return {k:{} for k in ["checkbox","combobox","slider","text"]}



        def readset_config(self):

            if self.config_file_exists():
            
                live = self.get_checkbox("live_update")
            
                self.set_checkbox("live_update", False)
           
                self.set_config(self.read_config())

                self.set_checkbox("live_update", live)
           
                print("Slider configuration,",self.get_combobox("nr_config"),"loaded.")

#                print("Slider configuration loaded. Plotting.")

#                self.plot()



        def getwrite_config(self):

            if self.must_save_config():
    
                out = self.get_config()

                with open(self.config_file(),"w") as f:

                    json.dump(out, f)


                print("Written slider configuration", self.get_combobox("nr_config"))


        def get_config(self):

            method = self.get_combobox("save_config")


            current = self.current_config()

            if method == "Overwrite":
               
                return current

            saved = self.read_config()


            if method == "Update and extend":

                return {T:{**saved[T],**current[T]} for T in current}


            for T in current:

                for k in list(current[T].keys()):
                    
                    if (method == "Add extras" and k in saved[T]) or (method == "Update existing" and k not in saved[T]):
                        
                        current[T].pop(k)
                         
                saved[T].update(current[T])

            return saved 
        

            







        def current_config(self):

            status = dict()

            status["checkbox"] = {c.objectName():c.isChecked() for c in self.findChildren(QtWidgets.QCheckBox)}

#            status["checkbox"].pop("remember_config")
            status["checkbox"].pop("live_update")




            status["combobox"] = {c.objectName():c.currentIndex() for c in self.findChildren(QtWidgets.QComboBox)}

            status["combobox"].pop("nr_config")
            status["combobox"].pop("save_config")



            status["slider"] = {c.objectName():c.value() for c in self.findChildren(QtWidgets.QSlider)}

            status["text"] = {c.objectName():c.text() for c in self.findChildren(QtWidgets.QLineEdit)}

            return status




        
        def set_fun(self,type_):

            return {"checkbox":self.set_checkbox,"combobox":self.set_combobox,"slider":self.set_slider,"text":self.set_text}[type_]


        def set_config(self,status):

            current_objects = self.current_config()

            for (type_,state_) in status.items():
            

                f = self.set_fun(type_)
            
                for (name, value) in state_.items():
                    if name in current_objects[type_]:

                        f(name, value)
                


	# ----------------------------------------- #
	# -------------- combo box ---------------- #
	# ----------------------------------------- #


        def add_combobox(self, vs=[], label=None, key=None, next_row=False, columnSpan=1, function="passive", vdiv=False):

            combo = QtWidgets.QComboBox(objectName=self.check_key(key,label))

            combo.addItems(list(map(str,vs)))
 
            self.connect_object(combo.currentTextChanged.connect,function)

            columnSpan = self.implement_newrow(next_row, vdiv, columnSpan, label is not None)

            self.add_label(label)

            self.add_widget(combo,columnSpan)

        def get_combobox(self,name):
            """Get the value of a combobox"""
            obj = self.findChild(QtWidgets.QComboBox,name)
            return obj.currentText()

        def set_combobox(self,name,v):
            """Get the value of a combobox"""
            self.findChild(QtWidgets.QComboBox,name).setCurrentIndex(v)

	# ----------------------------------------- #
	# -------------- slider ------------------- #
	# ----------------------------------------- #

        def add_slider(self,key=None,
                label=None,vs=range(100),v0=0,
                next_row=False,columnSpan=-1,function="passive",vdiv=False):


            key = self.check_key(key,label)

            slider = QtWidgets.QSlider(Qt.Horizontal,objectName=key)


            slider.vs = np.array(vs) 

            slider.setMinimum(0)
            slider.setTickInterval(1)
            slider.setMaximum(len(vs)-1)

            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

            if v0 is None: v0 = 0

            slider.setValue(v0)
 
            self.connect_object(slider.valueChanged.connect,function)

            columnSpan = self.implement_newrow(next_row,vdiv,columnSpan,(label is not None)+1)


            self.add_label(label)

            self.slidertext[key] = self.add_label(self.SliderVal(v0))

            self.add_widget(slider,columnSpan,max_width=False)

            self.setLayout(self.layout)




        def set_slider(self,name,v):

            self.findChild(QtWidgets.QSlider,name).setValue(v)

        def get_slider(self,name,index=False):
            """Get the value of a slider"""
            slider = self.findChild(QtWidgets.QSlider,name)
      

            ival = int(slider.value())

            val = slider.vs[ival]

            self.slidertext[name].setText(self.SliderVal(val))
           



            if index:
              return val,ival

            return val


	# ----------------------------------------- #
	# ----------- line edit (text) ------------ #
	# ----------------------------------------- #

        def add_text(self,key=None,label=None,text="",columnSpan=1,next_row=False,max_width=True,min_width=False,vdiv=False,function="passive"):

            le = QtWidgets.QLineEdit(objectName=self.check_key(key,label))

            le.setText(str(text))

            self.connect_object(le.editingFinished.connect,function)

            columnSpan = self.implement_newrow(next_row,vdiv,columnSpan,label is not None)

            self.add_label(label,max_width)

            self.add_widget(le,columnSpan,max_width,min_width)

            self.setLayout(self.layout)

        def get_text(self,name):
            le = self.findChild(QtWidgets.QLineEdit,name)
            return le.text() # return text

        def set_text(self,name,text):
            le = self.findChild(QtWidgets.QLineEdit,name)
            le.setText(str(text))


	# ----------------------------------------- #
	# ------------- check box ----------------- #
	# ----------------------------------------- #


        def add_checkbox(self,label=None,key=None,max_width=True,function="passive",vdiv=False,next_row=False,status=False):

            box = QtWidgets.QCheckBox(objectName=self.check_key(key,label))

            box.setChecked(status)

            self.connect_object(box.stateChanged.connect,function)

            columnSpan = self.implement_newrow(next_row,vdiv,1,label is not None)

            self.add_label(label,max_width)

            self.add_widget(box,1,max_width)

            self.setLayout(self.layout)

        def set_checkbox(self,name,state):

            self.findChild(QtWidgets.QCheckBox,name).setChecked(state)


        def get_checkbox(self,name):
            """Get the state of a checkbox"""

            obj = self.findChild(QtWidgets.QCheckBox,name)

            return obj.isChecked()


	# ----------------------------------------- #
	# ------------- button -------------------- #
	# ----------------------------------------- #

        def add_button(self,function,label=None,key=None,next_row=False,columnSpan=1,vdiv=False,max_width=True):

            btn = QtWidgets.QPushButton(label,objectName=self.check_key(key,label))


            self.connect_object(btn.clicked.connect,function)

            columnSpan = self.implement_newrow(next_row,vdiv,columnSpan)

            self.add_widget(btn,columnSpan,max_width)


	# ----------------------------------------- #
	# -------- update / save figure ----------- #
	# ----------------------------------------- #
     

        def add_functionalities(self):



            self.add_button(self.plot, label="Update plot", key="update", next_row=True)

            self.add_checkbox(label="Live", key="live_update", next_row=False, status=True, function=None)


            self.add_button(self.readset_config, label="Load config.", key="load_config",next_row=False, vdiv=False)


#            self.add_checkbox(label="Save config.", key="remember_config", next_row=False, status=False, function=None)


            self.add_combobox(["No", "Update and extend", "Overwrite", "Update existing", "Add extras"], key="save_config", label="Save config.", function=self.getwrite_config)

            self.add_combobox(range(1,6), #label="Save config.", 
                            key="nr_config", function=self.getwrite_config)

            self.add_button(self.save_file,label="Save file(s)",key="save_window",next_row=False,vdiv=True)

        

            self.add_text(key="save_path",text="",next_row=False,max_width=False,function=self.update_savebutton,vdiv=False,min_width=True,columnSpan=1)

            self.add_checkbox(function=self.update_savebutton,label="Replace?",key="save_replace",next_row=False,max_width=False,vdiv=False)

            

            resolutions = [str(n)+"MP" for n in [2,5,10,20]]



            self.add_combobox(['No','pdf']+resolutions, label="High res.", key="save_justfig", next_row=False, function=self.update_savebutton,vdiv=False)
 

#            self.add_button(self.set_savepath_minus,label="Fig.nr --",key="save_decrease",next_row=False,vdiv=True)
            self.add_button(self.set_savepath_plus,label="Fig.nr ++",key="save_increase",next_row=False,vdiv=False)

            self.set_savepath()
 
            self.update_savebutton()


        def set_savepath(self):

#            fn = str(self.figure_number).zfill(3) 

            fn = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


            folder = os.getcwd() + "/pyqt_figures/"  


            if os.path.isdir(folder)==False:

              os.mkdir(folder)

            path = folder + fn 

            self.set_text("save_path",path)
           
            self.update_savebutton()

            return path

        def update_savebutton(self):

            fn = self.get_text("save_path")

            isfiles = any([os.path.exists(fn+e) for e in ["_pyqt.png","_highres.pdf","_highres.png"]])

            replace = self.get_checkbox("save_replace")

            btn = self.findChild(QtWidgets.QPushButton,"save_window")

            btn.setEnabled( replace or not isfiles)



        def set_savepath_plus(self):

            self.figure_number = min(999, self.figure_number+1)

            self.set_savepath()

        def set_savepath_minus(self):

            self.figure_number = max(0, self.figure_number-1)

            self.set_savepath()

        def save_file(self):

            fn = self.get_text("save_path") # without extension

            print("\nFilename:",fn) 

#            isfile = os.path.exists(fn)

            screen = QApplication.primaryScreen()

            screenshot = screen.grabWindow( self.winId() )

            S = [screenshot.width(), screenshot.height()]

            print("Screenshot resolution:",resol_str(S)) 

            screenshot.save(fn+"_pyqt.png",'png')	#,quality=100)
            
          

            justfig = self.get_combobox("save_justfig")
             
            if justfig != "No": 

              FN = fn+'_highres'

              if "pdf" in justfig:
  
                self.figure.savefig(FN+'.pdf', format='pdf') 

                print("Savefig: pdf")

              else: 

                dpi,S = calc_dpi(self.figure.get_size_inches(), justfig[:-2])

                print("Savefig resolution:", resol_str(S))
                        
                self.figure.savefig(FN+'.png',dpi=dpi,format='png') 


                os.system(f"convert {FN}.png {FN}.pdf; rm {FN}.png")



              print("Two files saved.\n")

            else:

              print("File saved.\n")

#            self.set_savepath()

            self.update_savebutton()

	# ----------------------------------------- #
        
    app = QApplication(sys.argv)

    main = Window()


    main.add_functionalities()


    return app,main



#===========================================================================#
#
#   Figure class which is called by the user
#
#---------------------------------------------------------------------------#



class Figure:


  def __init__(self,funfig,nrows=1,ncols=1,tight=True,**kwargs):


    def funfig_(obj):
 
      fig_, axes = plt.subplots(nrows,ncols,num=obj.figure.number,**kwargs) 

      funfig(obj,fig_,axes)

      if tight:

        with warnings.catch_warnings():  

          warnings.filterwarnings("ignore",category=UserWarning)

          fig_.tight_layout(h_pad=0.1,w_pad=0.1) # adjust axis

      return fig_
    
    self.app,self.main = get_interface(funfig_)


  def show(self):  


    self.main.plot()


    self.main.show()


    sys.exit(self.app.exec_())

    

  def add_slider(self,*args,**kwargs):

    if self.main.findChild(QtWidgets.QSlider,kwargs["key"]) is None:

      self.main.add_slider(*args,**kwargs)


  def add_text(self,*args,**kwargs):

    if self.main.findChild(QtWidgets.QLineEdit,kwargs["key"]) is None:

      self.main.add_text(*args,**kwargs)

  def set_text(self,*args):

      self.main.set_text(*args)


  def add_combobox(self,*args,**kwargs):

    if self.main.findChild(QtWidgets.QComboBox,kwargs["key"]) is None:
 
      self.main.add_combobox(*args,**kwargs)


  def add_checkbox(self,*args,**kwargs):
    if self.main.findChild(QtWidgets.QCheckBox,kwargs["key"]) is None:

      self.main.add_checkbox(*args,**kwargs)

  def new_row(self):

      self.main.jump_on_next_row()






if __name__ == '__main__':

  def funfig(obj,Fig,ax): 
  
  
    xs = np.linspace(0.,10.,300) # xgrid
  
  
    # get the value of the slider
    ys = np.cos(xs*obj.get_slider("k") + obj.get_slider("phi")) 
  
    ys = ys + float(obj.get_text("dy")) # shift the values
  
  
    ax.plot(xs,ys,c=obj.get_combobox("c")) # plot data
  
    ax.set_ylim([-2,2])
  
  
  fig = Figure(funfig) # initialize figure instance
  
  
  ks = np.linspace(1.0,3.0,50) # wavevectors
  ps = np.linspace(0.0,2.0,50)*np.pi # phases
  
  
  fig.add_slider(label="Wavevector",key="k",vs=ks)#,columnSpan=1)
  
  #  main.add_slider(label="Useless slider",key="us",vs=ks,next_row=False)
  
  fig.add_slider(label="Phi",key="phi",vs=ps)

#  fig.add_slider(label="Phi2",key="phi2",vs=ps,next_row=True,columnSpan=10)
#  
#  fig.add_text(label="Shift2",key="dy2",text="0.0",columnSpan=1,next_row=False)
#
#  fig.add_text(label="Shift3",key="dy3",text="0.0",columnSpan=2,next_row=False,vdiv=True)

  fig.add_text(label="Shift",key="dy",text="0.0",columnSpan=1)
  
  fig.add_combobox(["red","blue","black"],label="Color",key="c")
  
  
  
  
  
  fig.show()
  
  
  
