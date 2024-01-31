
from __future__ import division
from __future__ import print_function

import kivy
kivy.require('1.8.0')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.clock import Clock

class Demo00Ui(Widget):
    counter = 0

    def update(self, dt):
        self.counter += 1
        self.ids.f_counter.text = str(self.counter)

class Demo00App(App):
    def build(self):
        ui = Demo00Ui()
        Clock.schedule_interval(ui.update, 1/60)
        return ui

if __name__ == '__main__':
    Demo00App().run()

