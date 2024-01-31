#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QTimer>
#include <QHash>
#include <QList>
#include <QMainWindow>
#include "../clientlib/bbclientlib.h"

extern usbDevice_t *usbdev;

namespace Ui {
	class MainWindow;
}

class Oszi : public QWidget
{
	Q_OBJECT

public:
	Oszi(QWidget *parent = 0);

	int samples_len;
	int posttrig_samples;
	QHash<QString, QList<uint8_t> > samples_data;

protected:
	void paintEvent(QPaintEvent * event);
};

class MainWindow : public QMainWindow
{
	Q_OBJECT
public:
	MainWindow(QWidget *parent = 0);
	~MainWindow();

protected:
	bool ch_digital[4];
	bool ch_analog[4];
	int samplelen;

	int stim_len;
	uint8_t stim_outmask;
	uint8_t *stim_pwm;
	uint8_t *stim_digi;

	QTimer *timer_poll;
	void updateOszi();

protected slots:
	void update_clkinfo();
	void action_autorestart();
	void action_start_stop();
	void action_poll();
	void action_start_stop_toolsmenu(bool checked);
	void action_stim_open();
	void action_stim_reload();
	void action_stim_clear();

private:
	Ui::MainWindow *ui;
	Oszi *oszi;
};

#endif // MAINWINDOW_H
