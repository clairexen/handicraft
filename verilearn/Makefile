
# world: yosys/build.ok

emsdk_portable/build.ok:
	rm -rf emsdk_portable
	wget https://s3.amazonaws.com/mozilla-games/emscripten/releases/emsdk-portable.tar.gz
	tar xvzf emsdk-portable.tar.gz
	mv emsdk-portable.tar.gz emsdk_portable
	+cd emsdk_portable && ./emsdk update
	+cd emsdk_portable && ./emsdk install latest
	+cd emsdk_portable && ./emsdk activate latest
	+cd emsdk_portable && ./emsdk_env.sh
	touch emsdk_portable/build.ok

yosys/build.ok: emsdk_portable/build.ok
	rm -rf yosys
	git clone https://github.com/cliffordwolf/yosys.git
	make -C yosys config-emcc
	. emsdk_portable/emsdk_set_env.sh && $(MAKE) -C yosys
	touch yosys/build.ok

