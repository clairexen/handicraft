
CC = gcc
CXX = g++
CXXFLAGS += -MD -Os -ggdb -Wall -Wextra
LDLIBS += -lvisa -lreadline -lstdc++
USE_ARCHIVED := 1

rigol-ds2000-shell: rigol-ds2000-shell.o

install: rigol-ds2000-shell
	sudo install rigol-ds2000-shell /usr/local/bin/

# see http://www.librevisa.org/
install-librevisa:
	rm -rf vxi librevisa
ifeq ($(USE_ARCHIVED),1)
	tar xvjf download-archive.tbz2 --xform 's,^archive/,,' archive/librevisa
	tar xvjf download-archive.tbz2 --xform 's,^archive/,,' archive/vxi
else
	git clone http://www.librevisa.org/git/librevisa.git
	git clone http://www.librevisa.org/git/vxi.git
endif
	sed -i 's,2.69,2.68,' vxi/configure.ac
	cd vxi && bash ../librevisa/autogen.sh
	cd vxi && ./configure CFLAGS=-fPIC && make -j4
	cd vxi && sudo make install
	cd librevisa && bash autogen.sh
	cd librevisa && ./configure && make -j4
	cd librevisa && sudo make install
	sudo ldconfig

# see http://freneticrapport.blogspot.co.at/2013/07/raspberry-pi-rigol-ds2072-200mhz.html
# see http://www.eevblog.com/forum/testgear/sniffing-the-rigol's-internal-i2c-bus/msg296744/#msg296744
install-rigol-4in1-keygen:
	rm -rf miracl && mkdir miracl
ifeq ($(USE_ARCHIVED),1)
	tar xvjf download-archive.tbz2 --xform 's,^archive/miracl-,miracl/,' archive/miracl-master.zip
else
	cd miracl && wget https://github.com/CertiVox/MIRACL/archive/master.zip
endif
	cd miracl && unzip -j -aa -L master.zip
	cd miracl && if uname -m | grep -q 64; then bash -x linux64; else bash -x linux; fi
	$(CC) -o rigol-4in1-keygen rigol-4in1-keygen.c -Imiracl miracl/miracl.a
	sudo install rigol-4in1-keygen /usr/local/bin/

ifneq ($(USE_ARCHIVED),1)
archive:
	rm -rf archive && mkdir -p archive
	cd archive && git clone http://www.librevisa.org/git/librevisa.git
	cd archive && git clone http://www.librevisa.org/git/vxi.git
	cd archive && wget -O miracl-master.zip https://github.com/CertiVox/MIRACL/archive/master.zip
	cd archive && wget -O rigol-ds2000-keygen.c http://pastebin.com/raw.php?i=c1jrPy3y
	rm -rf archive/*/.git
	tar cvjf download-archive.tbz2 archive/
	rm -rf archive/
endif

doc:
	wget 'http://www.rigol.com/download/Oversea/DS/Programming_guide/DS2000_ProgrammingGuide_EN.pdf'

clean:
	rm -rf vxi librevisa miracl rigol-ds2000-keygen rigol-ds2000-keygen.c *.o *.d

mrproper: clean
	rm -rf DS2000_ProgrammingGuide_EN.pdf rigol-ds2000-shell

