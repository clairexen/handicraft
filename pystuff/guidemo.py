#!/usr/bin/env python3

import sys
from PySide import QtCore, QtGui

class Project:
    def __init__(self):
        self.prjname = "Unnamed.prj"
        self.files = dict()

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("GUI Demo")

        ## Project Toolbar

        self.projectToolbar = self.addToolBar("Project")

        self.newProjectAct = QtGui.QAction("&New Project", self, triggered=self.newProject)
        self.projectToolbar.addAction(self.newProjectAct)

        self.saveProjectAct = QtGui.QAction("&Save Project", self, triggered=self.saveProject)
        self.projectToolbar.addAction(self.saveProjectAct)

        self.openProjectAct = QtGui.QAction("&Open Project", self, triggered=self.openProject)
        self.projectToolbar.addAction(self.openProjectAct)

        self.projectToolbar.addSeparator()

        self.setFilesAct = QtGui.QAction("Set &Files", self, triggered=self.setFiles)
        self.projectToolbar.addAction(self.setFilesAct)

        self.saveAllAct = QtGui.QAction("&Save All", self, triggered=self.saveAll)
        self.projectToolbar.addAction(self.saveAllAct)

        self.configureAct = QtGui.QAction("&Configure", self, triggered=self.configure)
        self.projectToolbar.addAction(self.configureAct)

        self.runAct = QtGui.QAction("&Run", self, triggered=self.run)
        self.projectToolbar.addAction(self.runAct)

    def newProject(self):
        pass

    def openProject(self):
        pass

    def saveProject(self):
        pass

    def setFiles(self):
        pass

    def saveAll(self):
        pass

    def configure(self):
        pass

    def run(self):
        pass

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

