/////////////////////////////////////////////////////////////////////
////                                                             ////
////  WISHBONE rev.B2 compliant I2C Master controller Testbench  ////
////                                                             ////
////                                                             ////
////  Author: Richard Herveille                                  ////
////          richard@asics.ws                                   ////
////          www.asics.ws                                       ////
////                                                             ////
////  Downloaded from: http://www.opencores.org/projects/i2c/    ////
////                                                             ////
/////////////////////////////////////////////////////////////////////
////                                                             ////
//// Copyright (C) 2001 Richard Herveille                        ////
////                    richard@asics.ws                         ////
////                                                             ////
//// This source file may be used and distributed without        ////
//// restriction provided that this copyright statement is not   ////
//// removed from the file and that any derivative work contains ////
//// the original copyright notice and the associated disclaimer.////
////                                                             ////
////     THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY     ////
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED   ////
//// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS   ////
//// FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR      ////
//// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,         ////
//// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    ////
//// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE   ////
//// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR        ////
//// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  ////
//// LIABILITY, WHETHER IN  CONTRACT, STRICT LIABILITY, OR TORT  ////
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT  ////
//// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ////
//// POSSIBILITY OF SUCH DAMAGE.                                 ////
////                                                             ////
/////////////////////////////////////////////////////////////////////

//  CVS Log
//
//  $Id: tst_bench_top.v,v 1.8 2006-09-04 09:08:51 rherveille Exp $
//
//  $Date: 2006-09-04 09:08:51 $
//  $Revision: 1.8 $
//  $Author: rherveille $
//  $Locker:  $
//  $State: Exp $
//
// Change History:
//               $Log: not supported by cvs2svn $
//               Revision 1.7  2005/02/27 09:24:18  rherveille
//               Fixed scl, sda delay.
//
//               Revision 1.6  2004/02/28 15:40:42  rherveille
//               *** empty log message ***
//
//               Revision 1.4  2003/12/05 11:04:38  rherveille
//               Added slave address configurability
//
//               Revision 1.3  2002/10/30 18:11:06  rherveille
//               Added timing tests to i2c_model.
//               Updated testbench.
//
//               Revision 1.2  2002/03/17 10:26:38  rherveille
//               Fixed some race conditions in the i2c-slave model.
//               Added debug information.
//               Added headers.
//

`include "timescale.v"

module tst_bench_top();

	//
	// wires && regs
	//
	reg  clk;
	reg  rst;

	wire [31:0] adr;
	wire [ 7:0] dat_i, dat_o, dat0_i, dat1_i;
	wire we;
	wire stb;
	wire cyc;
	wire ack0, ack1;
	wire inta;

	reg [7:0] q, qq;

	wire scl, scl0_o, scl0_oen, scl1_o, scl1_oen;
	wire sda, sda0_o, sda0_oen, sda1_o, sda1_oen;

	parameter PRER_LO = 3'b000;
	parameter PRER_HI = 3'b001;
	parameter CTR     = 3'b010;
	parameter RXR     = 3'b011;
	parameter TXR     = 3'b011;
	parameter CR      = 3'b100;
	parameter SR      = 3'b100;

	parameter TXR_R   = 3'b101; // undocumented / reserved output
	parameter CR_R    = 3'b110; // undocumented / reserved output

	parameter RD      = 1'b1;
	parameter WR      = 1'b0;
	parameter SADR    = 7'b0010_000;

	//
	// Module body
	//

	// generate clock
	always #5 clk = ~clk;

	// hookup wishbone master model
	wb_master_model #(8, 32) u0 (
		.clk(clk),
		.rst(rst),
		.adr(adr),
		.din(dat_i),
		.dout(dat_o),
		.cyc(cyc),
		.stb(stb),
		.we(we),
		.sel(),
		.ack(ack0 || ack1),
		.err(1'b0),
		.rty(1'b0)
	);

	wire stb0 = stb & ~adr[3];
	wire stb1 = stb &  adr[3];

	assign dat_i = ({{8'd8}{stb0}} & dat0_i) | ({{8'd8}{stb1}} & dat1_i);

	// hookup wishbone_i2c_master core
	i2c_master_top i2c_top (

		// wishbone interface
		.wb_clk_i(clk),
		.wb_rst_i(1'b0),
		.arst_i(rst),
		.wb_adr_i(adr[2:0]),
		.wb_dat_i(dat_o),
		.wb_dat_o(dat0_i),
		.wb_we_i(we),
		.wb_stb_i(stb0),
		.wb_cyc_i(cyc),
		.wb_ack_o(ack0),
		.wb_inta_o(inta),

		// i2c signals
		.scl_pad_i(scl),
		.scl_pad_o(scl0_o),
		.scl_padoen_o(scl0_oen),
		.sda_pad_i(sda),
		.sda_pad_o(sda0_o),
		.sda_padoen_o(sda0_oen)
	),
	i2c_top2 (

		// wishbone interface
		.wb_clk_i(clk),
		.wb_rst_i(1'b0),
		.arst_i(rst),
		.wb_adr_i(adr[2:0]),
		.wb_dat_i(dat_o),
		.wb_dat_o(dat1_i),
		.wb_we_i(we),
		.wb_stb_i(stb1),
		.wb_cyc_i(cyc),
		.wb_ack_o(ack1),
		.wb_inta_o(inta),

		// i2c signals
		.scl_pad_i(scl),
		.scl_pad_o(scl1_o),
		.scl_padoen_o(scl1_oen),
		.sda_pad_i(sda),
		.sda_pad_o(sda1_o),
		.sda_padoen_o(sda1_oen)
	);


	// hookup i2c slave model
	i2c_slave_model #(SADR) i2c_slave (
		.scl(scl),
		.sda(sda)
	);

        // create i2c lines
	delay m0_scl (scl0_oen ? 1'bz : scl0_o, scl),
	      m1_scl (scl1_oen ? 1'bz : scl1_o, scl),
	      m0_sda (sda0_oen ? 1'bz : sda0_o, sda),
	      m1_sda (sda1_oen ? 1'bz : sda1_o, sda);

	pullup p1(scl); // pullup scl line
	pullup p2(sda); // pullup sda line

	integer cycles = 0;
	always @(posedge clk) cycles <= cycles+1;

	initial
	  begin
	      `ifdef WAVES
	         $shm_open("waves");
	         $shm_probe("AS",tst_bench_top,"AS");
	         $display("INFO: Signal dump enabled ...\n\n");
	      `endif

//	      force i2c_slave.debug = 1'b1; // enable i2c_slave debug information
	      force i2c_slave.debug = 1'b0; // disable i2c_slave debug information

	      $display("\nstatus: %10t Testbench started\n\n", $time);

	      $dumpfile("simtrace.fst");
	      // $dumpvars(0, tst_bench_top);
	      // $dumpports(tst_bench_top.i2c_top);
              $dumpvars(1, tst_bench_top.i2c_top.wb_clk_i);
	      $dumpvars(1, tst_bench_top.i2c_top.wb_rst_i);
	      $dumpvars(1, tst_bench_top.i2c_top.arst_i);
	      $dumpvars(1, tst_bench_top.i2c_top.wb_adr_i);
	      $dumpvars(1, tst_bench_top.i2c_top.wb_dat_i);
	      $dumpvars(1, tst_bench_top.i2c_top.wb_dat_o);
	      $dumpvars(1, tst_bench_top.i2c_top.wb_we_i);
	      $dumpvars(1, tst_bench_top.i2c_top.wb_stb_i);
	      $dumpvars(1, tst_bench_top.i2c_top.wb_cyc_i);
	      $dumpvars(1, tst_bench_top.i2c_top.wb_ack_o);
	      $dumpvars(1, tst_bench_top.i2c_top.wb_inta_o);
	      $dumpvars(1, tst_bench_top.i2c_top.scl_pad_i);
	      $dumpvars(1, tst_bench_top.i2c_top.scl_pad_o);
	      $dumpvars(1, tst_bench_top.i2c_top.scl_padoen_o);
	      $dumpvars(1, tst_bench_top.i2c_top.sda_pad_i);
	      $dumpvars(1, tst_bench_top.i2c_top.sda_pad_o);
	      $dumpvars(1, tst_bench_top.i2c_top.sda_padoen_o);
	      // $dumpports(tst_bench_top.i2c_top2);
              $dumpvars(1, tst_bench_top.i2c_top2.wb_clk_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.wb_rst_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.arst_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.wb_adr_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.wb_dat_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.wb_dat_o);
	      $dumpvars(1, tst_bench_top.i2c_top2.wb_we_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.wb_stb_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.wb_cyc_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.wb_ack_o);
	      $dumpvars(1, tst_bench_top.i2c_top2.wb_inta_o);
	      $dumpvars(1, tst_bench_top.i2c_top2.scl_pad_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.scl_pad_o);
	      $dumpvars(1, tst_bench_top.i2c_top2.scl_padoen_o);
	      $dumpvars(1, tst_bench_top.i2c_top2.sda_pad_i);
	      $dumpvars(1, tst_bench_top.i2c_top2.sda_pad_o);
	      $dumpvars(1, tst_bench_top.i2c_top2.sda_padoen_o);

	      // initially values
	      clk = 0;

	      // reset system
	      rst = 1'b0; // assert reset
	      repeat(2) @(negedge clk);
	      rst = 1'b1; // de-assert reset

	      $display("status: %10t done reset", $time);

	      @(posedge clk);

	      //
	      // program core
	      //

	      // program internal registers
	      u0.wb_write(1, PRER_LO, 8'hfa); // load prescaler lo-byte
	      u0.wb_write(1, PRER_LO, 8'hc8); // load prescaler lo-byte
	      u0.wb_write(1, PRER_HI, 8'h00); // load prescaler hi-byte
	      $display("status: %10t programmed registers", $time);

	      u0.wb_cmp(0, PRER_LO, 8'hc8); // verify prescaler lo-byte
	      u0.wb_cmp(0, PRER_HI, 8'h00); // verify prescaler hi-byte
	      $display("status: %10t verified registers", $time);

	      u0.wb_write(1, CTR,     8'h80); // enable core
	      $display("status: %10t core enabled", $time);

	      //
	      // access slave (write)
	      //

	      // drive slave address
	      u0.wb_write(1, TXR, {SADR,WR} ); // present slave address, set write-bit
	      u0.wb_write(0, CR,      8'h90 ); // set command (start, write)
	      $display("status: %10t generate 'start', write cmd %0h (slave address+write)", $time, {SADR,WR} );

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(0, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // send memory address
	      u0.wb_write(1, TXR,     8'h01); // present slave's memory address
	      u0.wb_write(0, CR,      8'h10); // set command (write)
	      $display("status: %10t write slave memory address 01", $time);

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(0, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // send memory contents
	      u0.wb_write(1, TXR,     8'ha5); // present data
	      u0.wb_write(0, CR,      8'h10); // set command (write)
	      $display("status: %10t write data a5", $time);

while (scl) #1;
force scl= 1'b0;
#100000;
release scl;

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // send memory contents for next memory address (auto_inc)
	      u0.wb_write(1, TXR,     8'h5a); // present data
	      u0.wb_write(0, CR,      8'h50); // set command (stop, write)
	      $display("status: %10t write next data 5a, generate 'stop'", $time);

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      //
	      // delay
	      //
//	      #100000; // wait for 100us.
//	      $display("status: %10t wait 100us", $time);

	      //
	      // access slave (read)
	      //

	      // drive slave address
	      u0.wb_write(1, TXR,{SADR,WR} ); // present slave address, set write-bit
	      u0.wb_write(0, CR,     8'h90 ); // set command (start, write)
	      $display("status: %10t generate 'start', write cmd %0h (slave address+write)", $time, {SADR,WR} );

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // send memory address
	      u0.wb_write(1, TXR,     8'h01); // present slave's memory address
	      u0.wb_write(0, CR,      8'h10); // set command (write)
	      $display("status: %10t write slave address 01", $time);

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // drive slave address
	      u0.wb_write(1, TXR, {SADR,RD} ); // present slave's address, set read-bit
	      u0.wb_write(0, CR,      8'h90 ); // set command (start, write)
	      $display("status: %10t generate 'repeated start', write cmd %0h (slave address+read)", $time, {SADR,RD} );

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // read data from slave
	      u0.wb_write(1, CR,      8'h20); // set command (read, ack_read)
	      $display("status: %10t read + ack", $time);

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // check data just received
	      u0.wb_read(1, RXR, qq);
	      if(qq !== 8'ha5)
	        $display("\nERROR: Expected a5, received %x at time %10t", qq, $time);
	      else
	        $display("status: %10t received %x", $time, qq);

	      // read data from slave
	      u0.wb_write(1, CR,      8'h20); // set command (read, ack_read)
	      $display("status: %10t read + ack", $time);

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // check data just received
	      u0.wb_read(1, RXR, qq);
	      if(qq !== 8'h5a)
	        $display("\nERROR: Expected 5a, received %x at time %10t", qq, $time);
	      else
	        $display("status: %10t received %x", $time, qq);

	      // read data from slave
	      u0.wb_write(1, CR,      8'h20); // set command (read, ack_read)
	      $display("status: %10t read + ack", $time);

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // check data just received
	      u0.wb_read(1, RXR, qq);
	      $display("status: %10t received %x from 3rd read address", $time, qq);

	      // read data from slave
	      u0.wb_write(1, CR,      8'h28); // set command (read, nack_read)
	      $display("status: %10t read + nack", $time);

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // check data just received
	      u0.wb_read(1, RXR, qq);
	      $display("status: %10t received %x from 4th read address", $time, qq);

	      //
	      // check invalid slave memory address
	      //

	      // drive slave address
	      u0.wb_write(1, TXR, {SADR,WR} ); // present slave address, set write-bit
	      u0.wb_write(0, CR,      8'h90 ); // set command (start, write)
	      $display("status: %10t generate 'start', write cmd %0h (slave address+write). Check invalid address", $time, {SADR,WR} );

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // send memory address
	      u0.wb_write(1, TXR,     8'h10); // present slave's memory address
	      u0.wb_write(0, CR,      8'h10); // set command (write)
	      $display("status: %10t write slave memory address 10", $time);

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	           u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      // slave should have send NACK
	      $display("status: %10t Check for nack", $time);
	      if(!q[7])
	        $display("\nERROR: Expected NACK, received ACK\n");

	      // read data from slave
	      u0.wb_write(1, CR,      8'h40); // set command (stop)
	      $display("status: %10t generate 'stop'", $time);

	      // check tip bit
	      u0.wb_read(1, SR, q);
	      while(q[1])
	      u0.wb_read(1, SR, q); // poll it until it is zero
	      $display("status: %10t tip==0", $time);

	      #250000; // wait 250us
	      $display("\n\nstatus: %10t Testbench done. Simulated a total of %1d clock cycles.", $time, cycles);
	      $finish;
	  end
endmodule

module delay (in, out);
  input  in;
  output out;

  assign out = in;

  specify
    (in => out) = (600,600);
  endspecify
endmodule


