#!/usr/bin/python

import sys
from PyQt4 import QtCore, QtGui

class MyMainwin(QtGui.QMainWindow):
	def __init__(self, parent = None):
		QtGui.QMainWindow.__init__(self, parent)

		self.mTest1 = self.menuBar().addMenu("Test 1")
		self.mTest1_a1 = self.mTest1.addAction("Do not click me!")
		self.connect(self.mTest1_a1, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("mTest1_a1_hdlr()"))

		self.wEditor = QtGui.QPlainTextEdit(self)
		self.setCentralWidget(self.wEditor)

	@QtCore.pyqtSlot()
	def mTest1_a1_hdlr(self):
		self.wEditor.appendPlainText("*** I said do not click, dumbass! ***")

app = QtGui.QApplication(sys.argv)

mainwin = MyMainwin()
mainwin.show()

app.exec_()

# Unfortunately it might segfault at program exit..
# see https://bugs.launchpad.net/ubuntu/+source/python-qt4/+bug/561303

