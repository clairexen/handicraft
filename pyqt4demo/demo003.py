
import sys, os
from math import *
from PyQt4 import QtCore, QtGui, QtOpenGL
from OpenGL import GL
from bsddb3.db import *
import dbxml

if not os.access("demo003.dbenv", os.F_OK):
    os.mkdir("demo003.dbenv")

dbenv = DBEnv()
dbenv.open("demo003.dbenv", DB_RECOVER|DB_REGISTER|DB_CREATE| \
        DB_INIT_LOCK|DB_INIT_LOG|DB_INIT_MPOOL|DB_INIT_TXN, 0);

mgr = dbxml.XmlManager(dbenv, 0)

try:
    container = mgr.openContainer("demo003.dbxml")
except dbxml.XmlException, inst:
    container = mgr.createContainer("demo003.dbxml")
    container.putDocument("content", """
        <lines>
            <line>
                <point x="-3" y="-3"/>
                <point x="-3" y="+3"/>
            </line>
            <line>
                <point x="-3" y="-3"/>
                <point x="+3" y="-3"/>
            </line>
            <line>
                <point x="+3" y="+3"/>
                <point x="+3" y="-2"/>
            </line>
            <line>
                <point x="+3" y="+3"/>
                <point x="-2" y="+3"/>
            </line>
        </lines>
    """, mgr.createUpdateContext())
    container.sync()

class MyGLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent = None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.rgb = (1.0, 0.0, 0.0)

    def initializeGL(self):
        return

    def paintGL(self):
        GL.glClearColor(0, 0, 0, 0);
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT);

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glLineWidth(2)
        GL.glColor3d(self.rgb[0], self.rgb[1], self.rgb[2])
        GL.glBegin(GL.GL_LINES)

        qc = mgr.createQueryContext()
        results = mgr.query("""
            for $i in doc("dbxml:demo003.dbxml/content")//line
                return string-join((string($i/point[1]/@x), string($i/point[1]/@y),
                                    string($i/point[2]/@x), string($i/point[2]/@y)), ",")
        """, qc)
        for i in results:
            p = i.asString().split(",")
            GL.glVertex3d(float(p[0]), float(p[1]), 0);
            GL.glVertex3d(float(p[2]), float(p[3]), 0);
        del results
        del qc

        GL.glEnd()
        return

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-5, +5, +5, -5, -10.0, +10.0)
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

        self.mTestAction = self.menuBar().addMenu("Test Actions")
        self.mTestAction1 = self.mTestAction.addAction("Action #1")
        self.mTestAction2 = self.mTestAction.addAction("Action #2")
        self.mTestAction3 = self.mTestAction.addAction("Action #3")
        self.mTestAction4 = self.mTestAction.addAction("Action #4")
        self.connect(self.mTestAction1, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("hTestAction1()"))
        self.connect(self.mTestAction2, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("hTestAction2()"))
        self.connect(self.mTestAction3, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("hTestAction3()"))
        self.connect(self.mTestAction4, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("hTestAction4()"))

        self.mRefresh = self.menuBar().addAction("Refresh")
        self.connect(self.mRefresh, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("hRefresh()"))

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

    @QtCore.pyqtSlot()
    def hTestAction1(self):
        mgr.query("""
            insert nodes <line><point x="-1" y="-1"/><point x="+1" y="+1"/></line>
                into doc("dbxml:demo003.dbxml/content")/lines
        """, mgr.createQueryContext())
        container.sync()
        self.wGlWidget.updateGL()

    @QtCore.pyqtSlot()
    def hTestAction2(self):
        mgr.query("""
            insert nodes (
                <line>
                    <point x="-4" y="-4"/>
                    <point x="-4" y="+4"/>
                </line>,
                <line>
                    <point x="-4" y="-4"/>
                    <point x="+4" y="-4"/>
                </line>,
                <line>
                    <point x="+4" y="+4"/>
                    <point x="+4" y="-2"/>
                </line>,
                <line>
                    <point x="+4" y="+4"/>
                    <point x="-2" y="+4"/>
                </line>
            ) into doc("dbxml:demo003.dbxml/content")/lines
        """, mgr.createQueryContext())
        container.sync()
        self.wGlWidget.updateGL()
        return

    @QtCore.pyqtSlot()
    def hTestAction3(self):
        lines = ""
        for i in range(0,60):
            j = i + 1
            lines = lines + '<line><point x="{x1}" y="{y1}"/><point x="{x2}" y="{y2}"/></line>'. \
                    format(x1=sin(2*pi*i/60)*4.5, y1=cos(2*pi*i/60)*4.5, x2=sin(2*pi*j/60)*4.5, y2=cos(2*pi*j/60)*4.5)
        mgr.query("""
            insert nodes <p>{lines}</p>/* into doc("dbxml:demo003.dbxml/content")/lines
        """.format(lines = lines), mgr.createQueryContext())
        container.sync()
        self.wGlWidget.updateGL()
        return

    @QtCore.pyqtSlot()
    def hTestAction4(self):
        mgr.query("""
            delete nodes doc("dbxml:demo003.dbxml/content")//line[position() mod 2=1]
        """, mgr.createQueryContext())
        container.sync()
        self.wGlWidget.updateGL()
        return

    @QtCore.pyqtSlot()
    def hRefresh(self):
        container.sync()
        self.wGlWidget.updateGL()
        return

app = QtGui.QApplication(sys.argv)

mainwin = MyMainwin()
mainwin.resize(500, 300)
mainwin.show()

app.exec_()

