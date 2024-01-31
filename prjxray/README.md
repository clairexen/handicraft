
Project X-Ray: Documenting the Xilinx 7-Series bit-stream format
================================================================

Project X-Ray aims at documenting the bit-stream format of Xilinx 7-Series FPGA
devices. The current focus is on the Clock Region X0Y2 of the xc7a50tfgg484-1
device, especially the large continous region of SLICEL fabric from
SLICE_X16Y100 to SLICE_X27Y149. However, the tools produced in the process are
expected to be fairly generic and should in work with all 7-Series FPGAs.

Note that all Xilinx 7-Series devices support partial reconfiguration. Thus it
is not neccessary to comprehensively document the entire bit-stream format in
order to do anything useful with it. Instead, a small region of the FPGA can be
isolated and treated like its own little FPGA embedded in a much larger FPGA.

Goals for this project are:

- Being able to analyse partial bit-streams and convert the bit-stream to a
  behavioral Verilog model of the configured FPGA. This will allow for
  building automatic regression tests to verify the correctness of the
  documentation: synthesis with Vivado followed convertion of bit-stream
  to Verilog and then formal verification of generated model against original
  Verilog code.

- Being able to generate partial bit-streams using a FOSS tool-chain (using
  Yosys, VPR, and a to-be-written bitstream writer).

- Ultimately it should be possible to run the FOSS tool-chain on the device
  (in a soft-core processor or, in case of Zynq, on an ARM core), generate
  partial bitstreams right there, and upload them into the FPGA frabric using
  the Internal Configuration Access Port (ICAP).


Quick Start Guide
-----------------

Prerequisites:

- The main development platform for Project X-Ray is Ubuntu, but as long as you
  are using any resonably up-to-date Linux OS you should be fine.

- Make sure you have Xilinx Vivado installed, and the `vivado` executable is in the PATH.

- Build the X-Ray tools: `cd tools; make`


Creating a database of LUT configuration bits:

- Edit the `settings.sh` file (set part name and tile region of interest).

- Make sure there is no `database/` directory with data from a previous run.

- Run the tilegrid fuzzer: `cd fuzzers/000-tilegrid; bash run.sh`

- Check the generated tile grid: `chromium-browser database/tilegrid.html`

- Run the lutbits fuzzer: `cd fuzzers/001-lutbits; bash run.sh`

- Generate HTML bit-stream reference: `cd htmlgen; bash run.sh`


Overall fuzzing methodology for Project X-Ray
---------------------------------------------

A series of scripts called *fuzzers* are executed to collect information about
configuration bits and their functions.

The tool `bitread` can read a Xilinx bit-stream file and generate an ASCII
representation of the bit-stream. All other tools in the X-Ray toolbox operate
in this ASCII representation.

The tool `bitmatch` can read an set of ASCII bit-stream file and a set of text
files specifying which bits are set or cleared in which of the bit-stream files.

Uniquely identifying N configuration bits in one batch with `bitmatch` requires
about `2*log(N)/log(2)` bit-stream files (assuming a random 50% chance for each
bit to be set or cleared in each bit-stream file).

When a fuzzer is done it writes a .cbit test file to the `database/` directory
that contains the gathered information.

Often quite a lot of experimentation is neccessary in order to get to a point
where the bit-stream format (and tools involved with generating the bit-stream)
is understood well enough so that a fuzzer can be written. The tools `bitgrep`
and `bitdiff` can be very useful during that phase.

Note that currently `bitread` can only process bit-streams with per-frame
checksums. Set `BITSTREAM.GENERAL.PERFRAMECRC` on your design to produce such
bit-streams.


Xilinx documents you should be familiar with:
---------------------------------------------

### UG470: 7 Series FPGAs Configuration User Guide

https://www.xilinx.com/support/documentation/user_guides/ug470_7Series_Config.pdf

*Chapter 5: Configuration Details* contains a good description of the overall
bit-stream format. (See section "Bitstream Composition" and following.)

### UG912: Vivado Design Suite Properties Reference Guide

http://www.xilinx.com/support/documentation/sw_manuals/xilinx2016_2/ug912-vivado-properties.pdf

Contains an excellent description of the in-memory data structures and
associated properties Vivado uses to describe the design and the chip. The TCL
interface provides a convenient interface to access this information.

### UG903: Vivado Design Suite User Guide: Using Constraints

http://www.xilinx.com/support/documentation/sw_manuals/xilinx2016_2/ug903-vivado-using-constraints.pdf

The fuzzers generate designs (HDL + Constraints) that use many physical
contraints constraints (placement and routing) to produce bit-streams with
exactly the desired features. It helps to learn about the available constraints
before starting to write fuzzers.

### UG901: Vivado Design Suite User Guide: Synthesis

http://www.xilinx.com/support/documentation/sw_manuals/xilinx2016_2/ug901-vivado-synthesis.pdf

*Chapter 2: Synthesis Attributes* contains an overview of the Verilog
attributes that can be used to control Vivado Synthesis. Many of them
are useful for writing fuzzer designs. There is some natural overlap
with UG903.


Other documentation that might be of use:
-----------------------------------------

Doc of .bit container file format:  
http://www.pldtool.com/pdf/fmt_xilinxbit.pdf

Open-Source Bitstream Generation for FPGAs, Ritesh K Soni, Master Thesis:  
https://vtechworks.lib.vt.edu/bitstream/handle/10919/51836/Soni_RK_T_2013.pdf

From the bitstream to the netlist, Jean-Baptiste Note and Éric Rannaud:  
http://www.fabienm.eu/flf/wp-content/uploads/2014/11/Note2008.pdf

Wolfgang Spraul's Spartan-6 (xc6slx9) project:  
https://github.com/Wolfgang-Spraul/fpgatools

Marek Vasut's Typhoon Cyclone IV project:
http://git.bfuser.eu/?p=marex/typhoon.git

XDL generator/imported for Vivado:
https://github.com/byuccl/tincr

