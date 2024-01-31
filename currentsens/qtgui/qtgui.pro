
TEMPLATE       = app
CONFIG        += qt
QT            += opengl

HEADERS       += qtgui.h
SOURCES       += qtgui.cpp

FORMS += mainwin.ui

HEADERS       += qextserialbase.h qextserialport.h
SOURCES       += qextserialbase.cpp qextserialport.cpp

unix:HEADERS  += posix_qextserialport.h
unix:SOURCES  += posix_qextserialport.cpp
unix:DEFINES  += _TTY_POSIX_

win32:HEADERS += win_qextserialport.h
win32:SOURCES += win_qextserialport.cpp
win32:DEFINES += _TTY_WIN_

