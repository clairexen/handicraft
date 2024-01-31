#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>

void setrts(int fd, int v)
{
	int arg = TIOCM_RTS;
	ioctl(fd, v ? TIOCMBIS : TIOCMBIC, &arg);
}

void setdtr(int fd, int v)
{
	int arg = TIOCM_DTR;
	ioctl(fd, v ? TIOCMBIS : TIOCMBIC, &arg);
}

int main(int argc, char **argv)
{
	char *tts = argc > 1 ? argv[1] : "/dev/tts/0";
	int fd = open(tts, O_RDWR);

	while (1)
	{
		char ch = 0;
		sleep(1);

		int cmd_fd = open("ledctrl.cmd", O_RDONLY);
		read(cmd_fd, &ch, 1);
		close(cmd_fd);

		if (ch == 'x' || ch == 'X') {
			break;
		} else
		if (ch == 'r' || ch == 'R') {
			setrts(fd, 0);
			setdtr(fd, 1);
		} else
		if (ch == 'g' || ch == 'G') {
			setrts(fd, 1);
			setdtr(fd, 0);
		} else {
			setrts(fd, 0);
			setdtr(fd, 0);
		}
	}

	unlink("ledctrl.cmd");
	close(fd);
	return 0;
}

