
TARGET = bblab
TEMPLATE = app

SOURCES += main.cpp mainwindow.cpp
HEADERS += mainwindow.h
FORMS   += mainwindow.ui

QMAKE_CXXFLAGS += -ggdb
LIBS += ../clientlib/bbclientlib.o ../clientlib/hiddata.o

win32 {
LIBS += -lhid -lsetupapi
}

unix {
LIBS += -lusb
}

