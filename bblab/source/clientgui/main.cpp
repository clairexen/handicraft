
#include <QApplication>
#include <QMessageBox>
#include "mainwindow.h"

usbDevice_t *usbdev;

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	usbdev = bblabOpen();

	if (usbdev == NULL) {
		QMessageBox::critical(NULL, "BBLab - Critical Error!",
				"Failed to open BBLab USB device!\n\n"
				"Make sure it is connected to the PC\n"
				"and you have permissions to access\n"
				"the device and try again.\n");
		return 1;
	}

	int usbdev_version = bblabParameter(usbdev, SYSPAR_VERSION);
	if (usbdev_version != BBLAB_VERSION) {
		QMessageBox::critical(NULL, "BBLab - Critical Error!",
				QString("The BBLab USB device protocol version does not\n"
				"match the protocol version for this program!\n\n"
				"USB Device Protocol Version: %1.%2\n"
				"Program Version: %3.%4\n").arg(
				QString::number(usbdev_version >> 8),
				QString::number(usbdev_version & 0xff),
				QString::number(BBLAB_VERSION >> 8),
				QString::number(BBLAB_VERSION & 0xff)));
		return 1;
	}

	MainWindow w;
	w.show();
	a.exec();

	usbhidCloseDevice(usbdev);

	return 0;
}

