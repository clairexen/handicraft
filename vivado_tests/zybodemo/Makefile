
VIVADO_VER     := 2013.4
VIVADO_BIN     := /opt/Xilinx/Vivado/$(VIVADO_VER)/bin/vivado
VIVADO_HLS_BIN := /opt/Xilinx/Vivado_HLS/$(VIVADO_VER)/bin/vivado_hls
VIVADO_ECLIPSE := /opt/Xilinx/SDK/2014.1/bin/loader -exec ../../../eclipse/lnx64.o/eclipse
VIVADO_XMD_BIN := /opt/Xilinx/SDK/2014.1/bin/xmd

all: demo001.bit demo002.bit demo003.bit demo004.bit demo005.bit demo006.elf


### Design Synthesis ###

demo001.bit: demo001.v demo001.xdc demo001.tcl
	$(VIVADO_BIN) -nojournal -log demo001.log -mode batch -source demo001.tcl

demo002.bit: demo002.v demo002.xdc demo002.tcl
	$(VIVADO_BIN) -nojournal -log demo002.log -mode batch -source demo002.tcl

demo003.bit: demo003.v demo003.xdc demo003.tcl
	$(VIVADO_BIN) -nojournal -log demo003.log -mode batch -source demo003.tcl

demo004.bit: demo004.xdc demo004.tcl demo004_generator_101.zip demo004_consumer_101.zip
	$(VIVADO_BIN) -nojournal -log demo004.log -mode batch -source demo004.tcl

demo005.bit: demo005.xdc demo005.tcl demo004_generator_101.zip demo005_hlsconsumer_101.zip
	$(VIVADO_BIN) -nojournal -log demo005.log -mode batch -source demo005.tcl

demo006.bit: demo006.xdc demo006.tcl demo006_minigpio_101.zip
	$(VIVADO_BIN) -nojournal -log demo006.log -mode batch -source demo006.tcl


### IP Core Synthesis ###

demo004_generator_101.zip: demo004_generator.v demo004_generator.tcl
	$(VIVADO_BIN) -nojournal -nolog -mode batch -source demo004_generator.tcl

demo004_consumer_101.zip: demo004_consumer.v demo004_consumer.tcl
	$(VIVADO_BIN) -nojournal -nolog -mode batch -source demo004_consumer.tcl

demo005_hlsconsumer_101.zip: demo005_hlsconsumer.c demo005_hlsconsumer.tcl
	rm -rf demo005_hlsconsumer_tmp
	$(VIVADO_HLS_BIN) -f demo005_hlsconsumer.tcl
	cp demo005_hlsconsumer_tmp/sol_1/impl/ip/clifford_at_demo005_hlsconsumer_1_01.zip $@

demo006_minigpio_101.zip: demo006_minigpio.v demo006_minigpio.tcl
	$(VIVADO_BIN) -nojournal -nolog -mode batch -source demo006_minigpio.tcl


### Firmware Build ###

demo006.elf: PATH := /opt/Xilinx/SDK/2014.1/gnu/arm/lin/bin/:$(PATH)

demo006.elf: demo006.bit
	rm -rf demo006.sdk/hw2 demo006.sdk/bsp demo006.sdk/app demo006.sdk/.metadata
	$(VIVADO_ECLIPSE) -nosplash -data demo006.sdk -application org.eclipse.ant.core.antRunner -buildfile demo006_script.xml
	rm demo006.sdk/app/src/helloworld.c && cp demo006_main.c demo006.sdk/app/src/main.c
	$(VIVADO_ECLIPSE) -nosplash -data demo006.sdk -application org.eclipse.cdt.managedbuilder.core.headlessbuild -build all
	cp demo006.sdk/app/Release/app.elf demo006.elf


### Upload to ZYBO ###

prog001: demo001.bit
	$(VIVADO_BIN) -nojournal -nolog -mode batch -source upload.tcl -tclargs demo001.bit

prog002: demo002.bit
	$(VIVADO_BIN) -nojournal -nolog -mode batch -source upload.tcl -tclargs demo002.bit

prog003: demo003.bit
	$(VIVADO_BIN) -nojournal -nolog -mode batch -source upload.tcl -tclargs demo003.bit

prog004: demo004.bit
	$(VIVADO_BIN) -nojournal -nolog -mode batch -source upload.tcl -tclargs demo004.bit

prog005: demo005.bit
	$(VIVADO_BIN) -nojournal -nolog -mode batch -source upload.tcl -tclargs demo005.bit

prog006: demo006.elf demo006.bit
	$(VIVADO_XMD_BIN) -tcl demo006_xmd.tcl


### Digilent Documentation Download ###

docs:
	rm -rf docs docs_new && mkdir docs_new
	cd docs_new && wget http://www.digilentinc.com/Data/Products/ZYBO/ZYBO_sch.pdf
	cd docs_new && wget http://www.digilentinc.com/Data/Products/ZYBO/ZYBO_RM_B_V6.pdf
	cd docs_new && wget http://www.digilentinc.com/Data/Products/ZYBO/ZYBO_def.zip
	cd docs_new && wget http://www.digilentinc.com/Data/Products/ZYBO/ZYBO_Master_xdc.zip
	# cd docs_new && wget http://www.digilentinc.com/Data/Products/ZYBO/zybo_base_system.zip
	cd docs_new && for zipfile in *.zip; do unzip $$zipfile; done && rm -f *.zip
	mv docs_new docs


### Cleanup ###

clean:
	rm -rf .Xil .hw .srcs docs_new
	rm -rf demo004_iprepo demo004_generator_tmp demo004_consumer_tmp demo005_iprepo
	rm -rf demo005_hlsconsumer_tmp demo006_iprepo demo006_minigpio_tmp
	rm -f usage_statistics_webtalk.html usage_statistics_webtalk.xml
	rm -f *.log *.jou vivado_pid*.* hs_err_pid*.* ps_clock_registers.log

mrproper: clean
	rm -rf docs demo006.sdk
	rm -f demo004_generator_101.zip demo004_consumer_101.zip
	rm -f demo005_hlsconsumer_101.zip demo006_minigpio_101.zip demo006.elf
	rm -f demo001.bit demo002.bit demo003.bit demo004.bit demo005.bit demo006.bit

