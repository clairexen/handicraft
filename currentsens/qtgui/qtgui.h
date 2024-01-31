
#ifndef QTGUI_H
#define QTGUI_H

#include <QApplication>
#include <QMainWindow>
#include <QMessageBox>
#include <QClipboard>
#include <QLineEdit>
#include <QGLWidget>
#include <QPainter>
#include <QThread>
#include <QAction>
#include <QMutex>
#include <QEvent>
#include <QTime>
#include <QPen>
#include <QDir>

#include <qextserialport.h>

class MainWindow;

class AdcView : public QGLWidget
{
        Q_OBJECT

public:
	struct {
		int median, center, sum, sum_median, sum_center, raw_samples;
		bool triggered;
	} samples[100];

	int samples_counter;

	QTime clock;
	double laser_time;

	int trigger_low, trigger_high;
	int triggered;

	class SerIoThread : public QThread
	{
	public:
		QString line;
		AdcView *adcView;
		void run();
	};

	QString ioLine, ioPortName;
	static QEvent::Type ioLineEvent;
	QMutex ioThreadMutex;
	SerIoThread *ioThread;
	QextSerialPort *ioPort;
	double ioUpdateInterval;

	MainWindow *mw;

	AdcView(QWidget *parent);
	~AdcView();

public slots:
	void openPort(QString port);
	void refreshTrigger();

protected:
	void readNewData();
	bool event(QEvent *e);
	void paintEvent(QPaintEvent *event);
};

#include <ui_mainwin.h>

class MainWindow : public QMainWindow
{
        Q_OBJECT

public:
	class OpenPortAction : public QAction {
	public:
		QString portName;
		OpenPortAction(QString p, QObject *parent) :
				QAction("Open " + p, parent), portName(p) { }
	};
	Ui_MainWindow ui;
	MainWindow();
	~MainWindow();
protected slots:
	void openPortTriggered(QAction *action);
	void resetClicked();
	void copyClicked();
};

#endif

