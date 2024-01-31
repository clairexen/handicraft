#!/bin/bash

download:
	git clone git://opencircuitdesign.com/qflow-1.0
	git clone git://opencircuitdesign.com/qrouter-1.2
	git clone git://opencircuitdesign.com/magic-8.0
	wget http://opencircuitdesign.com/magic/archive/timberwolf-6.3.5.tgz
	tar xvzf timberwolf-6.3.5.tgz

example:
	mkdir -p example
	cd example && wget http://opencircuitdesign.com/qflow/example/map9v3.v
	cd example && wget http://opencircuitdesign.com/qflow/example/load.tcl
	cd example && wget http://opencircuitdesign.com/qflow/example/map9v3.tcl
	cd example && wget http://opencircuitdesign.com/qflow/example/osu035_stdcells.gds2

build-magic:
	cd magic-8.0 && ./configure --prefix=$$PWD/instdir
	cd magic-8.0 && $(MAKE) && $(MAKE) install

build-timberwolf:
	cd timberwolf-6.3 && tcsh -c 'source .twrc && make Makefiles && make install_non_yale'

build-qrouter:
	cd qrouter-1.2 && ./configure --prefix=$$PWD/instdir
	cd qrouter-1.2 && $(MAKE) && $(MAKE) install

build-qflow:
	cd qflow-1.0 && ./configure --prefix=$$PWD/instdir --with-libdir=$$PWD/instdir/data --with-bindir=$$PWD/instdir/bin \
			--with-timberwolf=$$PWD/../timberwolf-6.3 --with-magic=$$PWD/../magic-8.0/instdir/bin --with-qrouter=$$PWD/../qrouter-1.2/instdir/bin
	cd qflow-1.0 && $(MAKE) && $(MAKE) install

build:
	$(MAKE) build-magic
	$(MAKE) build-timberwolf
	$(MAKE) build-qrouter
	$(MAKE) build-qflow

bin:
	mkdir -p bin
	cd bin && ln -s ../qflow-1.0/instdir/bin/* .
	cd bin && ln -s ../qrouter-1.2/instdir/bin/* .
	cd bin && ln -s ../magic-8.0/instdir/bin/* .

mrproper:
	rm -rf qflow-1.0 qrouter-1.2 magic-8.0 timberwolf-6.3
	rm -rf timberwolf-6.3.5.tgz bin example

