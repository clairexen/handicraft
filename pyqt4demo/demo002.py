#!/usr/bin/python

from __future__ import division

import sys
from PyQt4 import QtCore, QtGui, QtOpenGL
from OpenGL import GL

class MyGLWidget(QtOpenGL.QGLWidget):
	def __init__(self, parent = None):
		QtOpenGL.QGLWidget.__init__(self, parent)
		self.rgb = (1.0, 0.0, 0.0)

	def initializeGL(self):
		return

	def paintGL(self):
		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glLoadIdentity()

		GL.glLineWidth(5)
		GL.glBegin(GL.GL_LINES)
		GL.glColor3d(self.rgb[0], self.rgb[1], self.rgb[2])
		GL.glVertex3d(-0.3, -0.3, 0); GL.glVertex3d(+0.3, +0.3, 0)
		GL.glVertex3d(+0.3, -0.3, 0); GL.glVertex3d(-0.3, +0.3, 0)
		GL.glEnd()
		return

	def resizeGL(self, width, height):
		GL.glViewport(0, 0, width, height)
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glLoadIdentity()
		GL.glOrtho(-0.5, +0.5, +0.5, -0.5, -10.0, +10.0)
		return

class MyMainwin(QtGui.QMainWindow):
	def __init__(self, parent = None):
		QtGui.QMainWindow.__init__(self, parent)

		self.mColorSel = self.menuBar().addMenu("Color")
		self.mColorSelR = self.mColorSel.addAction("Red")
		self.mColorSelG = self.mColorSel.addAction("Green")
		self.mColorSelB = self.mColorSel.addAction("Blue")
		self.connect(self.mColorSelR, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("hColorSelR()"))
		self.connect(self.mColorSelG, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("hColorSelG()"))
		self.connect(self.mColorSelB, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("hColorSelB()"))

		self.wGlWidget = MyGLWidget(self)
		self.setCentralWidget(self.wGlWidget)

	@QtCore.pyqtSlot()
	def hColorSelR(self):
		self.wGlWidget.rgb = (1.0, 0.0, 0.0);
		self.wGlWidget.updateGL()

	@QtCore.pyqtSlot()
	def hColorSelG(self):
		self.wGlWidget.rgb = (0.0, 1.0, 0.0);
		self.wGlWidget.updateGL()

	@QtCore.pyqtSlot()
	def hColorSelB(self):
		self.wGlWidget.rgb = (0.0, 0.0, 1.0);
		self.wGlWidget.updateGL()

app = QtGui.QApplication(sys.argv)

mainwin = MyMainwin()
mainwin.show()

app.exec_()

# Unfortunately it might segfault at program exit..
# see https://bugs.launchpad.net/ubuntu/+source/python-qt4/+bug/561303

