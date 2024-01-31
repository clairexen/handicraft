#include <string.h>
#include <stdio.h>
#include <string>

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>

int main(int argc, char **argv)
{
	Fl_Window *window = new Fl_Window(300, 180);

	Fl_Box *label = new Fl_Box(20, 10, 260, 10, "This is a test!");
	label->box(FL_NO_BOX);
	label->labelsize(20);
	label->labelfont(FL_BOLD);
	label->labeltype(FL_SHADOW_LABEL);

	Fl_Box *box = new Fl_Box(20, 40, 260, 90, "Hello, World!");
	box->box(FL_UP_BOX);
	box->labelsize(36);
	box->labelfont(FL_BOLD | FL_ITALIC);
	box->labeltype(FL_SHADOW_LABEL);

	Fl_Button *button_ptr[5];
	char button_labels[5][20];
	for (int i = 0; i < 5; i++) {
		snprintf(button_labels[i], 20, "Button #%d", i);
		button_ptr[i] = new Fl_Button(30 + 50*i, 150, 40, 20, button_labels[i]);
		button_ptr[i]->labelsize(10);
		button_ptr[i]->labelfont(FL_BOLD);
	}

	window->resizable(box);
	window->end();
	window->show(argc, argv);

	Fl::run();
	delete window;

	return 0;
}

