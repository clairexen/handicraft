#ifndef THREADHELPER_H
#define THREADHELPER_H

#include <QThread>

class ThreadHelper : public QThread {
public:
    static void sleep(unsigned long secs) {
        QThread::sleep(secs);
    }

    static void msleep(unsigned long msecs){
        QThread::msleep(msecs);
    }

    static void usleep(unsigned long usecs) {
        QThread::usleep(usecs);
    }
};

#endif // THREADHELPER_H
