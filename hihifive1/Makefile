TARGET = hihifive1
C_SRCS += hihifive1.c
CFLAGS += -Og -fno-builtin-printf
CLEAN_OBJS += libwrap.a openocd_upload.log openocd_debug.log

BSP_BASE =  $(HOME)/Temp/freedom-e-sdk/bsp

include $(BSP_BASE)/env/common.mk
export PATH := $(BSP_BASE)/../toolchain/bin/:$(PATH)

upload: $(TARGET)
	$(BSP_BASE)/tools/openocd_upload.sh $(TARGET) $(BSP_BASE)/env/freedom-e300-hifive1/openocd.cfg

debug: $(TARGET)
	$(BSP_BASE)/../toolchain/bin/riscv32-unknown-elf-gdb $(TARGET) \
		-ex "target remote | $(BSP_BASE)/../toolchain/bin/openocd -c 'gdb_port pipe; log_output openocd_debug.log' \
		                          -f $(BSP_BASE)/env/freedom-e300-hifive1/openocd.cfg" -ex "load"

console:
	teletype /dev/ttyUSB1 115200

