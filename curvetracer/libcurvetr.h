#ifndef LIBCURVETR_H
#define LIBCURVETR_H

void libcurvetr_setup(const char *serdev);
void libcurvetr_shutdown();
void libcurvetr_getraw(int v[6]);
void libcurvetr_getvolt(double v[6]);
int libcurvetr_gotkey();
void libcurvetr_calib();

#endif /* LIBCURVETR_H */
