
#include <stdio.h>
#include <string.h>
#include <QMessageBox>
#include <QFileDialog>
#include <QTextStream>
#include <QPainter>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "ThreadHelper.h"

// AVR PIN   HW Channel    UI Channel
//
//     PB5            4             1
//     PB2            1             2
//     PB3            2             3
//     PB4            3             4

static int uich2hwch(int uich)
{
	switch (uich) {
		case CHANNEL_DIGI1: return CHANNEL_DIGI4;
		case CHANNEL_DIGI2: return CHANNEL_DIGI1;
		case CHANNEL_DIGI3: return CHANNEL_DIGI2;
		case CHANNEL_DIGI4: return CHANNEL_DIGI3;
		case CHANNEL_ANALOG1: return CHANNEL_ANALOG4;
		case CHANNEL_ANALOG2: return CHANNEL_ANALOG1;
		case CHANNEL_ANALOG3: return CHANNEL_ANALOG2;
		case CHANNEL_ANALOG4: return CHANNEL_ANALOG3;
		default: return uich;
	}
}

MainWindow::MainWindow(QWidget * parent): QMainWindow(parent), ui(new Ui::MainWindow)
{
	stim_len = 0;
	stim_outmask = 0;
	stim_pwm = NULL;
	stim_digi = NULL;

	ui->setupUi(this);
	connect(ui->start_stop, SIGNAL(pressed()), this, SLOT(action_start_stop()));

	timer_poll = new QTimer(this);
	timer_poll->setSingleShot(true);
	connect(timer_poll, SIGNAL(timeout()), this, SLOT(action_poll()));

	oszi = new Oszi(this);
	QVBoxLayout *oszi_layout = new QVBoxLayout(ui->groupBox_3);
	oszi_layout->addWidget(oszi);

	connect(ui->clkdiv, SIGNAL(valueChanged(int)), this, SLOT(update_clkinfo()));
	connect(ui->outclkdiv, SIGNAL(valueChanged(int)), this, SLOT(update_clkinfo()));
	update_clkinfo();

	connect(ui->pretrig_channel, SIGNAL(currentIndexChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->pretrig_mode, SIGNAL(currentIndexChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->pretrig_value, SIGNAL(valueChanged(int)), this, SLOT(action_autorestart()));

	connect(ui->trigger_channel, SIGNAL(currentIndexChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->trigger_mode, SIGNAL(currentIndexChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->trigger_value, SIGNAL(valueChanged(int)), this, SLOT(action_autorestart()));

	connect(ui->ch1, SIGNAL(currentIndexChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->ch2, SIGNAL(currentIndexChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->ch3, SIGNAL(currentIndexChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->ch4, SIGNAL(currentIndexChanged(int)), this, SLOT(action_autorestart()));

	connect(ui->clkdiv, SIGNAL(valueChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->outclkdiv, SIGNAL(valueChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->samplelen, SIGNAL(valueChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->trigger_init_ticks, SIGNAL(valueChanged(int)), this, SLOT(action_autorestart()));
	connect(ui->posttrig_nticks, SIGNAL(valueChanged(int)), this, SLOT(action_autorestart()));

	connect(ui->posttrig_nticks, SIGNAL(valueChanged(int)), this, SLOT(action_autorestart()));

	connect(ui->actionSampler_Config, SIGNAL(toggled(bool)), ui->config_sampler, SLOT(setVisible(bool)));
	connect(ui->actionStimulus_Config, SIGNAL(toggled(bool)), ui->config_stim, SLOT(setVisible(bool)));

	connect(ui->actionRunning, SIGNAL(toggled(bool)), this, SLOT(action_start_stop_toolsmenu(bool)));

	connect(ui->stim_open, SIGNAL(clicked()), this, SLOT(action_stim_open()));
	connect(ui->stim_reload, SIGNAL(clicked()), this, SLOT(action_stim_reload()));
	connect(ui->stim_clear, SIGNAL(clicked()), this, SLOT(action_stim_clear()));
}

MainWindow::~MainWindow()
{
	delete ui;
	delete timer_poll;
}

uint8_t combobox2channel(QComboBox *w)
{
	if (w->currentText() == "Digital 1")
		return CHANNEL_DIGI1;
	if (w->currentText() == "Digital 2")
		return CHANNEL_DIGI2;
	if (w->currentText() == "Digital 3")
		return CHANNEL_DIGI3;
	if (w->currentText() == "Digital 4")
		return CHANNEL_DIGI4;
	if (w->currentText() == "Analog 1")
		return CHANNEL_ANALOG1;
	if (w->currentText() == "Analog 2")
		return CHANNEL_ANALOG2;
	if (w->currentText() == "Analog 3")
		return CHANNEL_ANALOG3;
	if (w->currentText() == "Analog 4")
		return CHANNEL_ANALOG4;
	if (w->currentText() == "Output Idx")
		return CHANNEL_OUTIDX;
	return CHANNEL_NONE;
}

uint8_t combobox2mode(QComboBox *w)
{
	if (w->currentText() == "Less-Then")
		return TRIGMODE_LT;
	if (w->currentText() == "Less-Equal")
		return TRIGMODE_LE;
	if (w->currentText() == "Equal")
		return TRIGMODE_EQ;
	if (w->currentText() == "Greater-Equal")
		return TRIGMODE_GE;
	if (w->currentText() == "Greater-Then")
		return TRIGMODE_GT;
	if (w->currentText() == "Rising Edge (LT => GE)")
		return TRIGMODE_LT_GE;
	if (w->currentText() == "Falling Edge (GT => LE)")
		return TRIGMODE_GT_LE;
	return 0;
}

void ch2mode(QComboBox *w, bool &digital, bool &analog)
{
	digital = false;
	analog = false;

	if (w->currentText() == "Digital")
		digital = true;

	if (w->currentText() == "Analog")
		analog = true;

	if (w->currentText() == "Both") {
		digital = true;
		analog = true;
	}
}

void MainWindow::update_clkinfo()
{
	int dev_freq = bblabParameter(usbdev, SYSPAR_FREQUENCY);
	int clkdiv = ui->clkdiv->value();
	int freq = dev_freq / clkdiv;
	int us = 1000000 / freq;

	ui->clkinfo->setText(QString("Sample Frequency: %1 Hz, Intervall: %2 us").arg(QString::number(freq), QString::number(us)));

	clkdiv = ui->outclkdiv->value();
	if (clkdiv == 0)
		clkdiv = ui->clkdiv->value();
	freq = dev_freq / clkdiv;
	us = 1000000 / freq;

	int pfreq = 0, pus = 0;

	if (stim_len > 0) {
		pfreq = dev_freq / (clkdiv*stim_len);
		pus = 1000000 / pfreq;
	}

	ui->outclkinfo->setText(QString("Sample Frequency: %1 Hz (%2 us),  Pattern Frequency: %3 Hz (%4 us)").arg(
			QString::number(freq), QString::number(us), QString::number(pfreq), QString::number(pus)));
}

void MainWindow::action_start_stop_toolsmenu(bool checked)
{
	if (checked && ui->start_stop->text() == "Start")
		action_start_stop();

	if (!checked && ui->start_stop->text() == "Stop")
		action_start_stop();
}

void MainWindow::action_autorestart()
{
	int ch, mode;

	ch = combobox2channel(ui->pretrig_channel);
	if (ch >= CHANNEL_DIGI1 && ch <= CHANNEL_DIGI4) {
		ui->pretrig_value->setMaximum(1);
		mode = combobox2mode(ui->pretrig_mode);
		if (mode == TRIGMODE_LT_GE)
			ui->pretrig_value->setValue(1);
		if (mode == TRIGMODE_GT_LE)
			ui->pretrig_value->setValue(0);
	} else {
		ui->pretrig_value->setMaximum(254);
	}

	ch = combobox2channel(ui->trigger_channel);
	if (ch >= CHANNEL_DIGI1 && ch <= CHANNEL_DIGI4) {
		ui->trigger_value->setMaximum(1);
		mode = combobox2mode(ui->trigger_mode);
		if (mode == TRIGMODE_LT_GE)
			ui->trigger_value->setValue(1);
		if (mode == TRIGMODE_GT_LE)
			ui->trigger_value->setValue(0);
	} else {
		ui->trigger_value->setMaximum(254);
	}

	if (ui->start_stop->text() == "Stop") {
		ui->start_stop->setText("Start");
		action_start_stop();
	}
}

void MainWindow::action_start_stop()
{
	timer_poll->stop();

	bblabStateSet(usbdev, STATE_SHUTDOWN);
	while (bblabStateGet(usbdev) != STATE_HALT) {
            ThreadHelper::usleep(10000);
	}

	if (ui->start_stop->text() == "Stop") {
		ui->start_stop->setText("Start");
		ui->actionRunning->setChecked(false);
		return;
	}

	struct bblab_config_t cfg;
	int memsize = sizeof(cfg);

	memset(&cfg, 0, sizeof(cfg));

	cfg.clkdiv = ui->clkdiv->value();
	cfg.outclkdiv = ui->outclkdiv->value();

	cfg.pretrig_mode_channel = uich2hwch(combobox2channel(ui->pretrig_channel)) | combobox2mode(ui->pretrig_mode);
	cfg.pretrig_value = ui->pretrig_value->value();

	cfg.trigger_mode_channel = uich2hwch(combobox2channel(ui->trigger_channel)) | combobox2mode(ui->trigger_mode);
	cfg.trigger_value = ui->trigger_value->value();

	cfg.trigger_init_ticks = ui->trigger_init_ticks->value();
	cfg.posttrig_nticks = ui->posttrig_nticks->value();

	cfg.samplelen = ui->samplelen->value();
	samplelen = cfg.samplelen;

	ch2mode(ui->ch1, ch_digital[0], ch_analog[0]);
	ch2mode(ui->ch2, ch_digital[1], ch_analog[1]);
	ch2mode(ui->ch3, ch_digital[2], ch_analog[2]);
	ch2mode(ui->ch4, ch_digital[3], ch_analog[3]);

	if (ch_digital[0] || ch_digital[1] || ch_digital[2] || ch_digital[3]) {
		cfg.digisampleptr = memsize;
		memsize += (cfg.samplelen+1) >> 1;
	}

	for (int i = 0; i < 4; i++) {
		if (ch_analog[i]) {
			cfg.adcsampleptr[uich2hwch(i+1)-1] = memsize;
			memsize += cfg.samplelen;
		}
	}

	if (stim_digi) {
		cfg.outlen = stim_len;
		cfg.outputmask = stim_outmask;
		cfg.digioutptr = memsize;
		memsize += (stim_len+1) >> 1;
	}

	if (stim_pwm) {
		cfg.outlen = stim_len;
		cfg.pwmoutptr = memsize;
		memsize += stim_len;
	}

	int dev_memsize = bblabParameter(usbdev, SYSPAR_MEMSIZE);
	if (dev_memsize < memsize) {
		QMessageBox::critical(NULL, "BBLab - Critical Error!",
				QString("Configuration exceeds device memory!\n\n"
				"Memory needed for this config: %1 bytes\n"
				"Memory available on the device: %2 bytes\n").arg(
				QString::number(memsize), QString::number(dev_memsize)));
		return;
	}

	// fprintf(stderr, "\n----- mem usage: %3d / %3d bytes -----\n", memsize, dev_memsize);
	// bblabConfigDump(&cfg, "   ");
	// fprintf(stderr, "----------------------------------------\n");

	if (stim_digi)
		bblabMemSet(usbdev, cfg.digioutptr, stim_digi, (stim_len+1) >> 1);

	if (stim_pwm)
		bblabMemSet(usbdev, cfg.pwmoutptr, stim_pwm, stim_len);

	bblabConfigConvert(&cfg);
	bblabMemSet(usbdev, 0, &cfg, sizeof(cfg));

	bblabStateSet(usbdev, STATE_CONFIG);
	ui->start_stop->setText("Stop");
	ui->actionRunning->setChecked(true);

	timer_poll->start(ui->poll_init_delay->value());
}

Oszi::Oszi(QWidget *parent) : QWidget(parent), samples_len(0), posttrig_samples(0) { }

void Oszi::paintEvent(QPaintEvent *)
{
	QPainter p(this);
	p.fillRect(0, 0, width(), height(), Qt::black);

	p.setPen(Qt::yellow);
	int x = width() * (samples_len-posttrig_samples-1.5) / (samples_len-1);
	p.drawLine(x, 0, x, height());

	int yoff = 5;
	int ystep = 0;

	QList<QString> ids;
	ids.append("A1");
	ids.append("A2");
	ids.append("A3");
	ids.append("A4");
	ids.append("D1");
	ids.append("D2");
	ids.append("D3");
	ids.append("D4");

	QListIterator<QString> iter1(ids);
	while (iter1.hasNext()) {
		QString id = iter1.next();
		if (!samples_data.contains(id))
			continue;
		ystep++;
	}

	if (ystep == 0)
		return;

	ystep = (height()-5) / ystep;

	QListIterator<QString> iter2(ids);
	while (iter2.hasNext())
	{
		QString id = iter2.next();
		if (!samples_data.contains(id))
			continue;

		p.setPen(Qt::yellow);
		p.drawLine(0, yoff + ystep - 5, width(), yoff + ystep - 5);

		p.setPen(Qt::green);
		int x1 = 0;
		int y1 = yoff + (ystep-5) * (256 - samples_data[id][0]) / 256;
		if (samples_data[id][0] == 255)
			y1 = -1;
		for (int i = 1; i < samples_len; i++) {
			uint8_t val = samples_data[id][i];
			int x2 = width() * i / (samples_len-1);
			if (val == 0xff) {
				p.setPen(Qt::gray);
				if (i == samples_len-1)
					p.drawLine(x1, y1, x2, y1);
			} else {
				int y2 = yoff + (ystep-5) * (256 - val) / 256;
				if (y1 == -1)
					y1 = y2;
				p.drawLine(x1, y1, x2, y2);
				x1 = x2, y1 = y2;
				p.setPen(Qt::green);
			}
		}

		yoff += ystep;

		p.setPen(Qt::white);
		p.drawText(10, yoff - 10, id);
	}
}

bool isInvalidSample(struct bblab_config_t *cfg, int idx)
{
	for (int i = 0; i < 4; i++)
		if (cfg->collision_tick[i] < idx && idx <= (cfg->collision_tick[i] + cfg->collision_duration[i]))
			return true;
	return false;
}

void MainWindow::updateOszi()
{
	oszi->samples_len = 0;
	oszi->samples_data.clear();

	struct bblab_config_t cfg;
	bblabMemGet(usbdev, 0, &cfg, sizeof(cfg));
	bblabConfigConvert(&cfg);

	oszi->samples_len = samplelen;
	oszi->posttrig_samples = cfg.posttrig_nticks;

	if (ch_digital[0] || ch_digital[1] || ch_digital[2] || ch_digital[3])
	{
		int buffer_len = (oszi->samples_len+1) / 2;
		uint8_t buffer[buffer_len];
	
		bblabMemGet(usbdev, cfg.digisampleptr, buffer, buffer_len);

		for (int i = 0; i < 4; i++)
		{
			if (!ch_digital[i])
				continue;

			QString id = QString("D%1").arg(QString::number(i+1));

			for (int j = 1; j <= oszi->samples_len; j++) {
				int idx = (cfg.sampleidx+j) % oszi->samples_len;
				uint8_t value = buffer[idx / 2];
				if (idx % 2 != 0)
					value = value >> 4;
				value = (value & (1 << (uich2hwch(i+1)-1))) ? 254 : 0;
				if (isInvalidSample(&cfg, oszi->samples_len-j))
					value = 255;
				oszi->samples_data[id].append(value);
			}
		}
	}

	for (int i = 0; i < 4; i++)
	{
		if (!ch_analog[i])
			continue;

		QString id = QString("A%1").arg(QString::number(i+1));

		uint8_t buffer[oszi->samples_len];
		bblabMemGet(usbdev, cfg.adcsampleptr[uich2hwch(i+1)-1], buffer, oszi->samples_len);

		for (int j = 1; j <= oszi->samples_len; j++) {
			int idx = (cfg.sampleidx+j) % oszi->samples_len;
			uint8_t value = buffer[idx];
			if (isInvalidSample(&cfg, oszi->samples_len-j))
				value = 255;
			oszi->samples_data[id].append(value);
		}
	}

	oszi->update();
}

void MainWindow::action_poll()
{
	uint8_t state = bblabStateGet(usbdev);

	if (state == STATE_RECORD) {
		timer_poll->start(1000 / ui->poll_frequency->value());
		return;
	}

	if (state == STATE_STANDBY) {
		updateOszi();
		bblabStateSet(usbdev, STATE_RECORD);
		timer_poll->start(ui->poll_init_delay->value());
		return;
	}

	bblabStateSet(usbdev, STATE_SHUTDOWN);
	ui->start_stop->setText("Start");
	ui->actionRunning->setChecked(false);
}

void MainWindow::action_stim_open()
{
	QString new_filename = QFileDialog::getOpenFileName(this, "Open File", "", "CSV Files (*.csv)");
	ui->stim_filename->setText(new_filename);
	action_stim_reload();
}

void MainWindow::action_stim_clear()
{
	ui->stim_filename->setText("");
	action_stim_reload();
}

void MainWindow::action_stim_reload()
{
	QString info;

	stim_len = 0;
	stim_outmask = 0;
	if (stim_pwm)
		free(stim_pwm);
	stim_pwm = NULL;
	if (stim_digi)
		free(stim_digi);
	stim_digi = NULL;

	if (!ui->stim_filename->text().isEmpty())
	{
		QFile file(ui->stim_filename->text());
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			QMessageBox::critical(NULL, "BBLab - Critical Error!",
					QString("Can't open stimulus file `%1'!").arg(ui->stim_filename->text()));
			return;
		}

		QString text = QTextStream(&file).readAll();
		QStringList lines = text.split(QRegExp("[\r\n]+"));

		for (int i = lines.length()-1; i >= 0; i--) {
			if (lines[i].contains(QRegExp("[a-zA-Z0-9]")))
				break;
			lines.removeLast();
		}

		if (lines.length() < 2) {
			QMessageBox::critical(NULL, "BBLab - Critical Error!",
					QString("Stimulus file `%1' contains no records!").arg(ui->stim_filename->text()));
			return;
		}

		QStringList headers = lines[0].split(QRegExp("[;,]"));
		for (int i = 0; i < headers.length(); i++) {
			headers[i].remove(QRegExp("[^a-zA-Z0-9]"));
		}

		stim_len = lines.length() - 1;

		info += QString("Using file `%1'.\n").arg(ui->stim_filename->text());
		info += QString("Number of samples: %1, Number of channels: %2\n").
				arg(QString::number(stim_len), QString::number(headers.length()));

		for (int i = 0; i < headers.length(); i++)
		{
			if (headers[i] == "pwm" || headers[i] == "PWM") {
				info += QString("Col %1 is PWM data:").arg(QString::number(i+1));
				if (stim_pwm) {
					info += QString(" dup ignored!\n");
					continue;
				}
				stim_pwm = (uint8_t*)calloc(stim_len, 1);
				for (int j = 0; j < stim_len; j++) {
					QString record = lines[j+1].split(QRegExp("[;,]"))[i];
					record.remove(QRegExp("[^a-zA-Z0-9]"));
					stim_pwm[j] = record.toUInt();
					info += QString(" %1").arg(QString::number(stim_pwm[j]));
				}
				info += "\n";
				continue;
			}
			if (headers[i] == "d1" || headers[i] == "D1" ||
					headers[i] == "d2" || headers[i] == "D2" ||
					headers[i] == "d3" || headers[i] == "D3" ||
					headers[i] == "d4" || headers[i] == "D4") {
				int chan = 0;
				if (headers[i] == "d1" || headers[i] == "D1")
					chan = uich2hwch(1)-1;
				if (headers[i] == "d2" || headers[i] == "D2")
					chan = uich2hwch(2)-1;
				if (headers[i] == "d3" || headers[i] == "D3")
					chan = uich2hwch(3)-1;
				if (headers[i] == "d4" || headers[i] == "D4")
					chan = uich2hwch(4)-1;
				if (!stim_digi)
					stim_digi = (uint8_t*)calloc((stim_len+1)/2, 1);
				info += QString("Col %1 is D%2 data:").arg(QString::number(i+1), QString::number(chan+1));
				if ((stim_outmask & (1 << chan)) != 0) {
					info += QString(" dup ignored!\n");
					continue;
				}
				stim_outmask |= 1 << chan;
				for (int j = 0; j < stim_len; j++) {
					QStringList records = lines[j+1].split(QRegExp("[;,]"));
					int val = 0;
					if (records.length() > i) {
						QString record = records[i];
						record.remove(QRegExp("[^a-zA-Z0-9]"));
						val = record.toUInt() != 0;
					}
					info += QString(" %1").arg(QString::number(val));
					if (j % 2 == 0) {
						stim_digi[j / 2] |= val << chan;
					} else {
						stim_digi[j / 2] |= val << (chan + 4);
					}
				}
				info += "\n";
				continue;
			}

			info += QString("Col %1 is of unknown type `%2'!\n").arg(QString::number(i+1), headers[i]);
		}
	}

	ui->stim_info->document()->setPlainText(info);
	action_autorestart();
	update_clkinfo();
}

