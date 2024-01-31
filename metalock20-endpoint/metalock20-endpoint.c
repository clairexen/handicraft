// gcc -o metalock20-endpoint -Wall -Wextra -Os metalock20-endpoint.c

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <termios.h>

int main(int argc, char **argv)
{
	int serfd;
	FILE *serf;

	if (argc != 3) {
		fprintf(stderr, "Usage example: %s /dev/ttyS0 ./authorize.sh\n", argv[0]);
		return 1;
	}

restart:
	if ((serfd = open(argv[1], O_RDWR)) < 0) {
		fprintf(stderr, "%s: open of serial dev failed (%s)! restart in 1 second..\n", argv[0], strerror(errno));
		sleep(1);
		goto restart;
	}
	
	struct termios tio;
	memset(&tio, 0, sizeof(tio));
	if (tcgetattr(serfd, &tio) < 0) {
		fprintf(stderr, "%s: tcgetattr on serial dev failed (%s)! restart in 1 second..\n", argv[0], strerror(errno));
		close(serfd);
		sleep(1);
		goto restart;
	}
	cfsetospeed(&tio, B19200);
	if (tcsetattr(serfd, TCSANOW, &tio) < 0) {
		fprintf(stderr, "%s: tcgetattr on serial dev failed (%s)! restart in 1 second..\n", argv[0], strerror(errno));
		close(serfd);
		sleep(1);
		goto restart;
	}

	serf = fdopen(serfd, "w+");
	while (1) {
		char buffer[1024];
		if (fgets(buffer, 1024, serf) == NULL) {
			fprintf(stderr, "%s: read from serial dev failed! restart in 1 second..\n", argv[0]);
			fclose(serf);
			sleep(1);
			goto restart;
		}

		char command[2048];
		snprintf(command, 2048, "%s '%s'", argv[2], buffer);
		fprintf(stderr, "%s: running \"%s\"..\n", argv[0], command);

		int ch;
		FILE *p = popen(command, "r");
		while ((ch = fgetc(p)) >= 0) {
			fputc(ch, stderr);
			fputc(ch, serf);
		}
		fflush(serf);
		fclose(p);
	}

	/* never reached */
	return 0;
}

