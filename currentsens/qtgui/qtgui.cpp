
#include "qtgui.h"

QEvent::Type AdcView::ioLineEvent = QEvent::None;

AdcView::AdcView(QWidget *parent) : QGLWidget(parent)
{
	if (ioLineEvent == QEvent::None)
		ioLineEvent = (QEvent::Type)QEvent::registerEventType();

	memset(samples, 0, sizeof(samples));
	samples_counter = 0;

	clock.start();
	laser_time = 0;

	trigger_low = 10000;
	trigger_high = 10000;
	triggered = false;

	ioPort = NULL;
	ioThread = NULL;

	ioPortName = QString();
	ioUpdateInterval = -5;

#if 1
	ioThread = new SerIoThread();
	ioThread->adcView = this;
	ioThread->start();
#endif

	mw = NULL;
}

AdcView::~AdcView()
{
	if (ioThread) {
		ioThreadMutex.lock();
		ioThread->terminate();
		ioThread->wait();
		ioThreadMutex.unlock();
		delete ioThread;
	}

	if (ioPort)
		delete ioPort;
}

void AdcView::readNewData()
{
	ioThreadMutex.lock();
	if (ioLine.length() > 0) {
		double time = clock.restart() / 1000.0;
		QStringList list = ioLine.split(' ');
		ioLine = QString();
		if (triggered > 0)
			laser_time += time;
		if (ioUpdateInterval < -1)
			ioUpdateInterval += 1;
		else
		if (ioUpdateInterval < 0)
			ioUpdateInterval = time;
		else
			ioUpdateInterval = ioUpdateInterval*0.9 + time*0.1;
		samples[samples_counter].median = list[0].toInt();
		samples[samples_counter].center = list[1].toInt();
		samples[samples_counter].sum = list[2].toInt();
		samples[samples_counter].sum_median = list[3].toInt();
		samples[samples_counter].sum_center = list[4].toInt();
		for (int i=0; i<100; i++)
			samples[i].raw_samples = list[5+i].toInt();
		if (triggered > 0) {
			if (samples[samples_counter].sum > trigger_low)
                                triggered += int(time * 50);
			else
                                triggered -= int(time * 30);
		} else {
			if (samples[samples_counter].sum > trigger_high)
                                triggered += int(time * 50);
			else
                                triggered -= int(time * 30);
		}
		if (triggered > 100)
			triggered = 100;
		if (triggered < 0)
			triggered = 0;
		samples[samples_counter].triggered = triggered > 0;
		samples_counter = (samples_counter+1) % 100;
		if (mw) {
			mw->ui.m1->setText(list[0]);
			mw->ui.m2->setText(list[1]);
			mw->ui.m3->setText(list[2]);
			mw->ui.m4->setText(list[3]);
			mw->ui.m5->setText(list[4]);
			QString euro_total = QString::number(mw->ui.euro_min->text().toDouble() * laser_time / 60.0, 'f', 2);
			if (euro_total != mw->ui.euro_total->text())
				mw->ui.euro_total->setText(euro_total);
			double seconds = laser_time;
                        int minutes = int(laser_time / 60);
			seconds -= minutes * 60;
			int hours = minutes / 60;
			minutes -= hours*60;
			QString time_msg;
			time_msg.sprintf("%02d:%02d:%05.2f", hours, minutes, seconds);
			if (time_msg != mw->ui.laser_time->text())
				mw->ui.laser_time->setText(time_msg);
			mw->ui.trigger->setRange(0, 100);
			mw->ui.trigger->setValue(triggered);
			if (!ioPortName.isNull()) {
				if (ioUpdateInterval < 0)
					mw->statusBar()->showMessage(QString("Connected to %1...").arg(ioPortName));
				else
					mw->statusBar()->showMessage(QString("Connected to %1 (%2 updates / min).").
							arg(ioPortName, QString::number(60 / ioUpdateInterval, 'f', 2)));
			}
		}
		update();
	}
	ioThreadMutex.unlock();
}

bool AdcView::event(QEvent *e) 
{
	if (e->type() == ioLineEvent)
		readNewData();
	return QWidget::event(e);
}

void AdcView::refreshTrigger()
{
	if (mw) {
		trigger_low = mw->ui.trigger_low->text().toInt();
		trigger_high = mw->ui.trigger_high->text().toInt();
	}
}

void AdcView::SerIoThread::run()
{
	char ch;
	bool firstLine = true;
	line = "";
	while (1) {
		if (!adcView->ioPort) {
			line = "0 0 0 0 0";
			for (int i=0; i<100; i++)
				line += " 0";
			adcView->ioThreadMutex.lock();
			adcView->ioLine = line;
			adcView->ioThreadMutex.unlock();
			QCoreApplication::postEvent(adcView, new QEvent(adcView->ioLineEvent));
			QThread::msleep(500);
			firstLine = true;
			line = "";
			continue;
		}
		if (!adcView->ioPort->getChar(&ch)) {
			QThread::msleep(100);
			continue;
		}
		if (ch == '\n') {
			if (!firstLine) {
				adcView->ioThreadMutex.lock();
				adcView->ioLine = line;
				adcView->ioThreadMutex.unlock();
				QCoreApplication::postEvent(adcView, new QEvent(adcView->ioLineEvent));
			}
			firstLine = false;
			line = "";
		} else {
			line += QString(ch);
		}
	}
}

void AdcView::openPort(QString port)
{
	if (ioThread) {
		ioThreadMutex.lock();
		ioThread->terminate();
		ioThread->wait();
		ioThreadMutex.unlock();
		delete ioThread;
	}

	if (ioPort)
		delete ioPort;

	ioPortName = port;
	ioUpdateInterval = -5;

	ioPort = new QextSerialPort(port);
	ioPort->setBaudRate(BAUD38400);
	ioPort->open(QIODevice::ReadOnly|QIODevice::Unbuffered);

	ioThread = new SerIoThread();
	ioThread->adcView = this;
	ioThread->start();
}

void AdcView::paintEvent(QPaintEvent*)
{
	QPainter painter(this);
	painter.setRenderHint(QPainter::Antialiasing);

	double ws = painter.viewport().width() / 99.0;
	double hs2 = -painter.viewport().height() / 200.0;
	double hs5 = -painter.viewport().height() / 500.0;

	painter.fillRect(painter.viewport(), Qt::black);
	painter.translate(0, painter.viewport().height());

	QPen p0, p1, p2, p3, p4;

	p0.setBrush(QBrush(Qt::gray));
	p1.setBrush(QBrush(Qt::blue));
	p2.setBrush(QBrush(Qt::green));
	p3.setBrush(QBrush(Qt::red));
	p3.setWidth(2);
	p4.setBrush(QBrush(Qt::white));

	painter.setPen(p0);
        painter.drawLine(int((samples_counter-0.5)*ws), int(0*hs2), int((samples_counter-0.5)*ws), int(200*hs2));

	for (int i=0; i<99; i++) {
		if (samples[i].triggered)
                        painter.fillRect(int(i*ws), int(0), int(ws+1), int(10*hs2), QColor(255, 128, 128));
		//if (i != samples_counter-1) {
		//	painter.setPen(p0);
		//	painter.drawLine(i*ws, samples[i].median*hs2, (i+1)*ws, samples[i+1].median*hs2);
		//	painter.drawLine(i*ws, samples[i].center*hs2, (i+1)*ws, samples[i+1].center*hs2);
		//}
		painter.setPen(p4);
                painter.drawLine(int(i*ws), int(samples[i].raw_samples*hs2), int((i+1)*ws), int(samples[i+1].raw_samples*hs2));
		if (i != samples_counter-1) {
			//int sum_median1 = samples[i+0].sum_median > 480 ? 480 : samples[i+0].sum_median;
			//int sum_median2 = samples[i+1].sum_median > 480 ? 480 : samples[i+1].sum_median;
			//int sum_center1 = samples[i+0].sum_center > 480 ? 480 : samples[i+0].sum_center;
			//int sum_center2 = samples[i+1].sum_center > 480 ? 480 : samples[i+1].sum_center;
			int sum1 = samples[i+0].sum > 480 ? 480 : samples[i+0].sum;
			int sum2 = samples[i+1].sum > 480 ? 480 : samples[i+1].sum;
			// painter.setPen(p1);
                        // painter.drawLine(int(i*ws), int(sum_median1*hs5), int((i+1)*ws), int(sum_median2*hs5));
			// painter.setPen(p2);
                        // painter.drawLine(int(i*ws), int(sum_center1*hs5), int((i+1)*ws), int(sum_center2*hs5));
			painter.setPen(p3);
                        painter.drawLine(int(i*ws), int(sum1*hs5), int((i+1)*ws), int(sum2*hs5));
		}
	}
}

MainWindow::MainWindow()
{
	ui.setupUi(this);
	ui.adcview->mw = this;
	ui.adcview->refreshTrigger();
	connect(ui.trigger_low, SIGNAL(textChanged(QString)), ui.adcview, SLOT(refreshTrigger()));
	connect(ui.trigger_high, SIGNAL(textChanged(QString)), ui.adcview, SLOT(refreshTrigger()));
#ifdef Q_OS_WIN32
	ui.menu_sp->addAction(new OpenPortAction("COM1", this));
	ui.menu_sp->addAction(new OpenPortAction("COM2", this));
	ui.menu_sp->addAction(new OpenPortAction("COM3", this));
	ui.menu_sp->addAction(new OpenPortAction("COM4", this));
	ui.menu_sp->addAction(new OpenPortAction("COM5", this));
	ui.menu_sp->addAction(new OpenPortAction("COM6", this));
	ui.menu_sp->addAction(new OpenPortAction("COM7", this));
	ui.menu_sp->addAction(new OpenPortAction("COM8", this));
#else
	QDir d("/dev");
	QStringList l = d.entryList(QStringList() << "ttyS*" << "ttyUSB*", QDir::System|QDir::CaseSensitive, QDir::Name);
	for (int i=0; i<l.length(); i++)
		ui.menu_sp->addAction(new OpenPortAction("/dev/" + l[i], this));
#endif
	connect(ui.menu_sp, SIGNAL(triggered(QAction*)), this, SLOT(openPortTriggered(QAction*)));
	connect(ui.reset, SIGNAL(pressed()), this, SLOT(resetClicked()));
	connect(ui.copy, SIGNAL(pressed()), this, SLOT(copyClicked()));
	ui.euro_min->setFocus(Qt::OtherFocusReason);
	setWindowTitle("Current Sensor GUI");
}

MainWindow::~MainWindow()
{
}

void MainWindow::openPortTriggered(QAction *action)
{
	OpenPortAction *a = dynamic_cast<OpenPortAction*>(action);
	if (a) {
		statusBar()->showMessage("Connected to " + a->portName + "...");
		ui.adcview->openPort(a->portName);
	}
}

void MainWindow::resetClicked()
{
	QMessageBox msgBox;
	msgBox.setText("Resetting laster timer.");
	msgBox.setInformativeText("Do you want discard the current timer value?");
	msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
	msgBox.setDefaultButton(QMessageBox::Ok);
	if (msgBox.exec() == QMessageBox::Ok) {
		ui.adcview->laser_time = 0.0;
	}
}

void MainWindow::copyClicked()
{
	QString text =
		QString::number(ui.euro_min->text().toDouble(), 'f', 2) + QString("\t") +
		QString::number(ui.adcview->laser_time / 60.0, 'f', 2) + QString("\t") +
		QString::number(ui.euro_min->text().toDouble() * ui.adcview->laser_time / 60.0, 'f', 2) + QString("\n");
	QApplication::clipboard()->setText(text, QClipboard::Clipboard);
	QApplication::clipboard()->setText(text, QClipboard::Selection);
}

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	MainWindow mainWindow;
	if (argc >= 2 && argv[1][0] != 0 && argv[1][0] != '-')
		mainWindow.ui.adcview->openPort(argv[1]);
	if (argc >= 3 && argv[2][0] != 0 && argv[2][0] != '-')
		mainWindow.ui.trigger_low->setText(argv[2]);
	if (argc >= 4 && argv[3][0] != 0 && argv[3][0] != '-')
		mainWindow.ui.trigger_high->setText(argv[3]);
	mainWindow.show();
	return app.exec();
}

