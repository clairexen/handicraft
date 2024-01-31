
from fltk import *

def mklabel(str):
    Fl_Box(Fl_Group.current().x(), Fl_Group.current().y(),
           Fl_Group.current().w(), Fl_Group.current().h()).label(str)

class MainWindow(Fl_Window):

    def __init__(self):
        Fl_Window.__init__(self, 500, 500)
        self.label("demo004")

        self.tile = Fl_Tile(0, 100, 500, 400)
        self.tile.resizable(Fl_Box(50, 150, 400, 300));

        self.leftbar = Fl_Group(0, 100, 100, 400)
        self.leftbar.box(FL_DOWN_BOX)
        mklabel("Left\nSidebar");
        self.leftbar.end()

        self.maincanvas = Fl_Group(100, 100, 300, 300)
        self.maincanvas.box(FL_DOWN_BOX)
        mklabel("Main Canvas");
        self.maincanvas.end()

        self.console = Fl_Group(100, 400, 300, 100)
        self.console.box(FL_DOWN_BOX)
        mklabel("Console")
        self.console.end()

        self.rightbar = Fl_Group(400, 100, 100, 400)
        self.rightbar.box(FL_DOWN_BOX)
        mklabel("Right\nSidebar");
        self.rightbar.end()

        self.tile.end()
        self.resizable(self.tile)

    def resize(self, new_x, new_y, new_w, new_h):
        dim_wl = self.leftbar.w()
        dim_wr = self.rightbar.w()
        dim_hc = self.console.h()
        Fl_Window.resize(self, new_x, new_y, new_w, new_h)
        if (dim_wl + dim_wr + 100) < self.tile.w() and (dim_hc + 100) < self.tile.h():
            self.leftbar.resize(self.tile.x(), self.tile.y(), dim_wl, self.tile.h())
            self.maincanvas.resize(self.tile.x() + dim_wl, self.tile.y(), self.tile.w() - dim_wl - dim_wr, self.tile.h() - dim_hc)
            self.console.resize(self.tile.x() + dim_wl, self.tile.y() + self.tile.h() - dim_hc, self.tile.w() - dim_wl - dim_wr, dim_hc)
            self.rightbar.resize(self.tile.x() + self.tile.w() - dim_wr, self.tile.y(), dim_wr, self.tile.h());

mainwin = MainWindow()
mainwin.show();

Fl.run();

