
#ifndef ARDUINO_CONSOLE_H
#define ARDUINO_CONSOLE_H

#ifdef  __cplusplus
extern "C" {
#endif

extern int consoleInit(int baud);
extern int consoleAvailable();
extern int consoleGetChar();
extern void consolePutChar(int ch);
extern int consolePrint(const char *string);
extern int consolePrintf(const char *fmt, ...);
extern int consoleReadDecimal(const char *prompt);

#ifdef  __cplusplus
}
#endif

#endif /* ARDUINO_CONSOLE_H */

