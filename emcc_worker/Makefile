
EMCC_FLAGS := -std=c++11 -O3 --memory-init-file 0 -s NO_EXIT_RUNTIME=1 --bind

all: example.js

emsdk_portable/emsdk_set_env.sh:
	wget https://s3.amazonaws.com/mozilla-games/emscripten/releases/emsdk-portable.tar.gz
	tar xvzf emsdk-portable.tar.gz
	+cd emsdk_portable && ./emsdk install latest
	+cd emsdk_portable && ./emsdk activate latest
	+cd emsdk_portable && ./emsdk_env.sh

example.js: example.cc emsdk_portable/emsdk_set_env.sh
	source emsdk_portable/emsdk_set_env.sh && emcc $(EMCC_FLAGS) -o example.js example.cc

