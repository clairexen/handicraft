
all: test001.bin test002.bin test003.bin test004.bin test005.bin \
		test006.bin test007.bin test008.bin test009.bin
	$(MAKE) -C ../../utils
	../../utils/bindiff test*.bin

%.bin: %.v %.pcf
	icecuberun $(basename $<)

clean:
	rm -rf *.tmp *.bin

