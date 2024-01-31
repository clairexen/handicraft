
all: test000.bin test001.bin
	$(MAKE) -C ../../utils
	grep --color=never '^[^*]' test000.tmp/outputs/router/top.route
	grep --color=never '^[^*]' test001.tmp/outputs/router/top.route
	../../utils/bindiff test*.bin

%.bin: %.v %.pcf
	icecuberun $(basename $<)

%.pcf: test000.pcf
	ln -s test000.pcf $@

clean:
	rm -rf *.tmp *.bin

