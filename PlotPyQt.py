import numpy as np
import sys,os

from PyQt5.QtCore import Qt

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QGridLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


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
            self._dynamic_ax.clear()

            self.figure.clear()

            fig = plot_figure(self) # get that figure

#            plt.tight_layout(h_pad=0.1,w_pad=0.1) # adjust axis


            if fig.number!=self.figure.number: 
                raise ValueError("You must plot in the same figure as the one initialized for the interface, use fig = plt.figure(obj.figure.number) in your function, where obj is the input of your function")

            self._dynamic_ax.figure.canvas.draw()


        def update_plot(self):

            if self.get_checkbox("live_update"):
              self.plot()


        def Vdiv(self):
            div = QtWidgets.QLabel("")
            div.setStyleSheet("QLabel {background-color: #3e3e3e; padding: 0; margin: 0; border-bottom: 1 solid #666; border-top: 1 solid #2a2a2a;}")
            div.setMaximumWidth(2.5)
        
            return div

        def SliderVal(self,x):
          return str(np.round(x,3))


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

        def implement_newrow(self,next_row,vdiv,columnSpan):

            if next_row:
              self.row += 1 
              self.column = 0 

            else:
              columnSpan = max(1,columnSpan)

              if vdiv:
                self.layout.addWidget(self.Vdiv(),self.row,self.column)
                self.column +=1

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
	# -------------- combo box ---------------- #
	# ----------------------------------------- #


        def add_combobox(self,cs,label=None,key=None,next_row=True,columnSpan=-1,function="passive",vdiv=False):

            combo = QtWidgets.QComboBox(objectName=self.check_key(key,label))

            combo.addItems(list(map(str,cs)))
 
            self.connect_object(combo.currentTextChanged.connect,function)

            columnSpan = self.implement_newrow(next_row,vdiv,columnSpan)

            self.add_label(label)

            self.add_widget(combo,columnSpan)

        def get_combobox(self,name):
            """Get the value of a combobox"""
            obj = self.findChild(QtWidgets.QComboBox,name)
            return obj.currentText()



	# ----------------------------------------- #
	# -------------- slider ------------------- #
	# ----------------------------------------- #

        def add_slider(self,key=None,
                label=None,vs=range(100),v0=0,
                next_row=True,columnSpan=-1,function="passive",vdiv=False):


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

            columnSpan = self.implement_newrow(next_row,vdiv,columnSpan)

            self.add_label(label)

            self.slidertext[key] = self.add_label(self.SliderVal(v0))

            self.add_widget(slider,columnSpan,max_width=False)

            self.setLayout(self.layout)


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

        def add_text(self,key=None,label=None,text="",columnSpan=-1,next_row=True,max_width=True,min_width=False,vdiv=False,function="passive"):

            le = QtWidgets.QLineEdit(objectName=self.check_key(key,label))

            le.setText(str(text))

            self.connect_object(le.editingFinished.connect,function)

            columnSpan = self.implement_newrow(next_row,vdiv,columnSpan)

            self.add_label(label,max_width)

            self.add_widget(le,columnSpan,max_width,min_width)

            self.setLayout(self.layout)

        def get_text(self,name):
            le = self.findChild(QtWidgets.QLineEdit,name)
            return le.text() # return text

        def set_text(self,name,text):
            le = self.findChild(QtWidgets.QLineEdit,name)
            le.setText(text) 


	# ----------------------------------------- #
	# ------------- check box ----------------- #
	# ----------------------------------------- #


        def add_checkbox(self,label=None,key=None,next_row=True,columnSpan=-1,max_width=True,function="passive",vdiv=False,status=False):

            box = QtWidgets.QCheckBox(objectName=self.check_key(key,label))

            box.setChecked(status)

            self.connect_object(box.stateChanged.connect,function)

            columnSpan = self.implement_newrow(next_row,vdiv,columnSpan)

            self.add_label(label,max_width)

            self.add_widget(box,columnSpan,max_width)

            self.setLayout(self.layout)


        def get_checkbox(self,name):
            """Get the state of a checkbox"""

            obj = self.findChild(QtWidgets.QCheckBox,name)

            return obj.isChecked()


	# ----------------------------------------- #
	# ------------- button -------------------- #
	# ----------------------------------------- #

        def add_button(self,function,label=None,key=None,next_row=True,columnSpan=-1,vdiv=False,max_width=True):

            btn = QtWidgets.QPushButton(label,objectName=self.check_key(key,label))

            self.connect_object(btn.clicked.connect,function)

            columnSpan = self.implement_newrow(next_row,vdiv,columnSpan)

            self.add_widget(btn,columnSpan,max_width)


	# ----------------------------------------- #
	# -------- update / save figure ----------- #
	# ----------------------------------------- #
     

        def add_functionalities(self):


            self.add_button(self.plot,label="Update plot",key="update",columnSpan=1)



            self.add_checkbox(function=self.update_plot,label="Live update",key="live_update",columnSpan=1,next_row=False,status=True)






            self.add_button(self.save_file,label="Save file(s)",key="save_window",next_row=False,vdiv=True)
        


            self.add_text(key="save_path",text="",next_row=False,max_width=False,function=self.update_savebutton,vdiv=False,min_width=True)

            self.add_checkbox(function=self.update_savebutton,label="Replace?",key="save_replace",next_row=False,max_width=False,vdiv=True)

            

            self.add_combobox(['pdf']+list(map(str,range(200,601,100))),label="Higher resolution",key="save_justfig",next_row=False,function=None,vdiv=True)
 

            self.add_button(self.set_savepath_minus,label="Fig.nr --",key="save_decrease",next_row=False,vdiv=True)
            self.add_button(self.set_savepath_plus,label="Fig.nr ++",key="save_increase",next_row=False,vdiv=False)

            self.set_savepath()
 
            self.update_savebutton()


        def set_savepath(self):

            fn = str(self.figure_number).zfill(3)

            folder = os.getcwd() + "/pyqt_figures/"  

            path = folder + fn +".png"

            if os.path.isdir(folder)==False:

              os.mkdir(folder)



            self.set_text("save_path",path)
           
            self.update_savebutton()

            return path

        def update_savebutton(self):

            fn = self.get_text("save_path")

            isfiles = any([os.path.exists(fn+e) for e in ["","_plot.pdf","_plot.png"]])

            replace = self.get_checkbox("save_replace")

            btn = self.findChild(QtWidgets.QPushButton,"save_window")

            btn.setEnabled( replace or not isfiles)



        def set_savepath_plus(self):

            self.figure_number = min(999,self.figure_number+1)

            self.set_savepath()

        def set_savepath_minus(self):

            self.figure_number = max(0,self.figure_number-1)

            self.set_savepath()

        def save_file(self):

            fn = self.get_text("save_path")

            isfile = os.path.exists(fn)

            screen = QApplication.primaryScreen()

            screenshot = screen.grabWindow( self.winId() )

            screenshot.save(fn,'png')

            print("Files saved:",fn)
             
            justfig = self.get_combobox("save_justfig")

            if "pdf" in justfig:

              self.figure.savefig(fn+'_plot.pdf',format='pdf')

            else:
              self.figure.savefig(fn+'_plot.png',dpi=int(justfig),format='png') 

            self.update_savebutton()

        
    app = QApplication(sys.argv)

    main = Window()


    main.add_functionalities()


    return app,main

#if __name__ == '__main__':
#    def funfig(obj): # dummy function
#        xs = np.linspace(0.,10.,300) # xgrid
#        # get the value of the slider
#        ys = np.cos(xs*obj.get_slider("k") + obj.get_slider("phi")) 
#        ys = ys + float(obj.get_text("dy")) # shift the values
#
#
##        print("I drew")
#        fig = plt.figure(obj.figure.number) # initialize figure
#        fig.clear()
#        plt.plot(xs,ys,c=obj.get_combobox("c")) # plot data
#        plt.ylim([-2,2])
#        return fig 
#
#    app,main = get_interface(funfig) # get the interface
#
#    ks = np.linspace(1.0,3.0,50) # wavevectors
#    ps = np.linspace(0.0,2.0,50)*np.pi # phases
#
#    main.add_slider(label="Wavevector",key="k",vs=ks)#,columnSpan=1)
#
# 
##    main.add_slider(label="Useless slider",key="us",vs=ks,next_row=False)
#
#    main.add_slider(label="Phi",key="phi",vs=ps)
#
#    main.add_text(label="Shift",key="dy",text="0.0",columnSpan=1)
#
#    main.add_combobox(["red","blue","black"],label="Color",key="c",next_row=False)
#
#
#
#    main.plot()
#
#    main.show()
#
#
#    sys.exit(app.exec_())


class Figure:


  def __init__(self,funfig,nrows=1,ncols=1,tight=True,**kwargs):


    def funfig_(obj):
 
      fig_, axes = plt.subplots(nrows,ncols,num=obj.figure.number,**kwargs) 

      funfig(obj,fig_,axes)

      if tight:
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


#  def add_text(self,*args,**kwargs):
#    self.main.add_text(*args,**kwargs)

  def add_text(self,*args,**kwargs):

    if self.main.findChild(QtWidgets.QLineEdit,kwargs["key"]) is None:

      self.main.add_text(*args,**kwargs)

  def add_combobox(self,*args,**kwargs):


    if self.main.findChild(QtWidgets.QComboBox,kwargs["key"]) is None:
 
      self.main.add_combobox(*args,**kwargs)


  def add_checkbox(self,*args,**kwargs):

    if self.main.findChild(QtWidgets.QCheckBox,kwargs["key"]) is None:

      self.main.add_checkbox(*args,**kwargs)



