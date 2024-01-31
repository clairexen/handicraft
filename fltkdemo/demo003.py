
from fltk import *

def mklabel(str):
    Fl_Box(Fl_Group.current().x(), Fl_Group.current().y(), Fl_Group.current().w(), Fl_Group.current().h()).label(str)

def mkwin():
    win = Fl_Window(500, 500)
    win.label("demo003")

    tile = Fl_Tile(0, 100, 500, 400)

    gr1 = Fl_Group(0, 100, 100, 400)
    gr1.box(FL_DOWN_BOX)
    mklabel("Left\nSidebar");
    gr1.end()

    gr2 = Fl_Group(100, 100, 300, 300)
    gr2.box(FL_DOWN_BOX)
    mklabel("Main Canvas");
    gr2.end()

    console = Fl_Group(100, 400, 300, 100)
    console.box(FL_DOWN_BOX)
    mklabel("Console")
    console.end()

    gr3 = Fl_Group(400, 100, 100, 400)
    gr3.box(FL_DOWN_BOX)
    mklabel("Right\nSidebar");
    gr3.end()

    tp = Fl_Box(50, 150, 400, 300);
    tile.resizable(tp);

    tile.end()
    win.resizable(tile)
    win.show()
    return win

mainwin = mkwin()
Fl.run();

