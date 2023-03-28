from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
import sys
import time 
import datetime 
import numpy as np 

def mylongfunction(*args): 

    print(args)

    time.sleep(2)

class TheBoss(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TheBoss, self).__init__(parent)
        self.resize(300,200)
        self.VL = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel()
        self.VL.addWidget(self.label)
        self.logger = Logger() 

        self.logger.sec_signal.connect(self.label.setText)

        self.logger.start()

    def closeEvent(self,event):
        self.logger.terminate()

class Logger(QtCore.QThread):
    sec_signal = pyqtSignal(str)
    def __init__(self, parent=None):
        super(Logger, self).__init__(parent)

        self.initial_time = datetime.datetime.now()
        self.current_time = datetime.datetime.now() 

        self.go = True 

    def run(self):
        #this is a special fxn that's called with the start() fxn
        while self.go:
            time.sleep(0.5+np.random.rand())


            self.current_time = datetime.datetime.now()

            self.sec_signal.emit(str((self.current_time - self.initial_time).seconds)) 


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Thread Example")
    window = TheBoss()
    window.show()
    sys.exit(app.exec_())
