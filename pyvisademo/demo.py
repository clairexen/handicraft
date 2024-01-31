
import visa

dg1022_usb = visa.instrument("USB0::0x09C4::0x0400::DG1D150900596::INSTR")
print dg1022_usb.ask("*IDN?")

