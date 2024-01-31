
all: test000.bin test001.bin test002.bin test003.bin
	$(MAKE) -C ../../utils
	../../utils/bindiff test*.bin

%.bin: %.v %.pcf
	icecuberun $(basename $<)

clean:
	rm -rf *.tmp *.bin

