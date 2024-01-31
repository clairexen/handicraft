
#include <signal.h>
#include <termios.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <setjmp.h>

int fd;
FILE *f;
bool verbose;
struct termios tcattr_old;
const char *tts_name;
sigjmp_buf jbe;

void sigint_hdl(int dummy)
{
	siglongjmp(jbe, 1);
}

uint8_t serbuffer[1024];
int serbuffer_idx, serbuffer_len;
bool serbuffer_end_of_block;

int serialreadbyte()
{
	if (serbuffer_idx < serbuffer_len) {
		serbuffer_end_of_block = serbuffer_idx + 1 == serbuffer_len;
		return serbuffer[serbuffer_idx++];
	}

	int rc = read(fd, serbuffer, 1024);
	if (rc > 0) {
		serbuffer_idx = 0;
		serbuffer_len = rc;
		return serialreadbyte();
	}

	if (rc == 0)
		return -1;
	return -2;
}

unsigned char serialread()
{
	int ch = serialreadbyte();
	if (ch >= 0) {
		if (verbose) {
			if (32 < ch && ch < 127)
				printf("<0x%02x:'%c'>", ch, ch);
			else if (ch > 127)
				printf("<0x%02x:0b%d%d%d%d%d%d%d%d>", ch,
						(ch & 0x80) != 0, (ch & 0x40) != 0, (ch & 0x20) != 0, (ch & 0x10) != 0,
						(ch & 0x08) != 0, (ch & 0x04) != 0, (ch & 0x02) != 0, (ch & 0x01) != 0);
			else
				printf("<0x%02x>", ch);
			printf(serbuffer_end_of_block ? " EOB\n" : "\n");
		}
		return ch;
	}
	if (ch == -1)
		fprintf(stderr, "I/O Error on tts: EOF\n");
	else
		fprintf(stderr, "I/O Error on tts: %s\n", strerror(errno));
	tcsetattr(fd, TCSAFLUSH, &tcattr_old);
	exit(1);
}

void help(const char *progname)
{
	fprintf(stderr, "Usage: %s [-v] [-t ttydev] [-n bytes_per_line] [outputfile]\n", progname);
	exit(1);
}

int main(int argc, char **argv)
{
	int opt;
	int bytes_per_line = 1;
	const char *tts = "/dev/ttyACM0";

	while ((opt = getopt(argc, argv, "vt:n:")) != -1) {
		switch (opt) {
		case 'v':
			verbose = true;
			break;
		case 't':
			tts = optarg;
			break;
		case 'n':
			bytes_per_line = atoi(optarg);
			if (bytes_per_line <= 0)
				help(argv[0]);
			break;
		default:
			help(argv[0]);
		}
	}

	if (optind+1 != argc && optind != argc)
		help(argv[0]);

	if (optind+1 == argc) {
		f = fopen(argv[optind], "w");
		if (f == NULL) {
			fprintf(stderr, "Failed to open output file `%s': %s\n", argv[optind], strerror(errno));
			exit(1);
		}
	} else {
		f = stdout;
	}

	
	fd = open(tts, O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "Failed to open tts device `%s': %s\n", tts, strerror(errno));
		exit(1);
	}
	
	tcgetattr(fd, &tcattr_old);
	struct termios tcattr = tcattr_old;
	tcattr.c_iflag = IGNBRK | IGNPAR;
	tcattr.c_oflag = 0;
	tcattr.c_cflag = CS8 | CREAD | CLOCAL;
	tcattr.c_lflag = 0;
	cfsetspeed(&tcattr, B2000000);
	tcsetattr(fd, TCSAFLUSH, &tcattr);


	struct timeval tv_start, tv_stop;
	gettimeofday(&tv_start, NULL);

	size_t bytes = 0;
	if (sigsetjmp(jbe, 1) == 0)
	{
		signal(SIGINT, &sigint_hdl);
		while (1) {
			for (int i = 0; i < bytes_per_line; i++, bytes++) {
				fprintf(f, "%d%s", serialread(), i+1 == bytes_per_line ? "\n" : "\t");
				if (bytes == 0)
					gettimeofday(&tv_start, NULL);
				if (bytes % 1024 == 0)
					write(2, ".", 1);
			}
		}
	}

	gettimeofday(&tv_stop, NULL);
	double tv_diff = (tv_stop.tv_sec - tv_start.tv_sec) + 1e-6*(tv_stop.tv_usec - tv_start.tv_usec);

	fprintf(stderr, "Transfered %zd bytes in %.2f seconds. Avg. rate: %.2f kB/s.\n",
			bytes, tv_diff, 1e-3 * bytes / tv_diff);
	fclose(f);

	return 0;
}
