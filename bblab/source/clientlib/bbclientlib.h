
#ifndef _BBCLIENTLIB_H_
#define _BBCLIENTLIB_H_

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "hiddata.h"
#include "../firmware/bblab_common.h"

usbDevice_t *bblabOpen(void);

void bblabMemSet(usbDevice_t *dev, int addr, const void *data, int len);
void bblabMemGet(usbDevice_t *dev, int addr, void *data, int len);
int bblabParameter(usbDevice_t *dev, int parameter);
int bblabStateGet(usbDevice_t *dev);
void bblabStateSet(usbDevice_t *dev, int state);

void bblabConfigConvert(struct bblab_config_t *cfg);
void bblabConfigDump(const struct bblab_config_t *cfg, const char *prefix);

#ifdef __cplusplus
}
#endif

#endif /* _BBCLIENTLIB_H_ */

