
`timescale 1ns / 1ps

module SINCPDE_DIV(clk, N, D, Q, SYNC_IN, SYNC_OUT);

input clk;
input signed [47:0] N, D;
output signed [17:0] Q;

input SYNC_IN;
output reg SYNC_OUT;

(* fsm_encoding="one-hot", safe_implementation="no" *)
integer state;

reg negdiv;
reg [4:0] bitcount;

reg [47:0] dsp_ab;
reg [47:0] dsp_c;
wire [47:0] dsp_p;

assign Q = dsp_p;

reg [2:0] dsp_mode;
(* rom_extract = "no" *) reg [3:0] dsp_alumode;
(* rom_extract = "no" *) reg [6:0] dsp_opmode;

always @* begin
	case (dsp_mode)
		0: begin
			dsp_opmode <= 7'b 000_11_00;
			dsp_alumode <= 4'b 0000;
		end
		1: begin
			dsp_opmode <= 7'b 000_11_00;
			dsp_alumode <= 4'b 0011;
		end
		2: begin
			dsp_opmode <= 7'b 000_00_11;
			dsp_alumode <= 4'b 0000;
		end
		3: begin
			dsp_opmode <= 7'b 000_00_11;
			dsp_alumode <= 4'b 0011;
		end
		4: begin
			dsp_opmode <= 7'b 010_00_00;
			dsp_alumode <= 4'b 0000;
		end
		5: begin
			dsp_opmode <= 7'b 010_11_00;
			dsp_alumode <= 4'b 0011;
		end
		6: begin
			dsp_opmode <= dsp_p[47] ? 7'b 010_11_00 : 7'b 010_00_00;
			dsp_alumode <= 4'b 0000;
		end
		default: begin
			dsp_opmode <= 'bx;
			dsp_alumode <= 'bx;
		end
	endcase
end

always @(posedge clk) begin
	SYNC_OUT <= 0;
	dsp_mode <= 'bx;

	if (SYNC_IN) begin
		dsp_ab <= N;
		dsp_c <= D <<< 4;
		negdiv <= N[47] != D[47];
		bitcount <= 17;
		state <= 110;
	end else
	case (state)
		110: begin
			bitcount <= bitcount - 1;
			if ((|dsp_ab[47:43] == &dsp_ab[47:43]) && (|dsp_c[47:43] == &dsp_c[47:43])) begin
				dsp_ab <= dsp_ab << 4;
				dsp_c <= dsp_c << 4;
			end else begin
				dsp_mode <= dsp_c[47] ? 1 : 0;
				state <= 120;
			end
		end
		120: begin
			dsp_mode <= dsp_c[47] ? 1 : 0;
			bitcount <= bitcount - 1;
			if (bitcount == 0) begin
				bitcount <= 17;
				state <= 130;
			end
		end
		130: begin
			dsp_c <= dsp_p;
			dsp_mode <= dsp_ab[47] ? 3 : 2;
			dsp_ab <= 0;
			state <= 140;
		end
		140: begin
			dsp_mode <= 4;
			state <= 150;
		end
		150: begin
			dsp_mode <= 5;
			if (bitcount != 17)
				dsp_ab <= (dsp_ab << 1) | !dsp_p[47];
			state <= 160;
		end
		160: begin
			dsp_mode <= 6;
			dsp_c <= dsp_c >> 1;
			bitcount <= bitcount - 1;
			state <= bitcount != 0 ? 150 : 170;
		end
		170: begin
			dsp_mode <= negdiv ? 3 : 2;
			state <= 180;
		end
		180: begin
			SYNC_OUT <= 1;
			state <= 190;
		end
	endcase
end

/********************************* BEGIN TEMPLATE: xilinx_dsp48e1.v *********************************/

// Cascade: 30-bit (each) output: Cascade Ports
wire [29:0] DSP_ACOUT;                               // 30-bit output: A port cascade output
wire [17:0] DSP_BCOUT;                               // 18-bit output: B port cascade output
wire        DSP_CARRYCASCOUT;                        // 1-bit output: Cascade carry output
wire        DSP_MULTSIGNOUT;                         // 1-bit output: Multiplier sign cascade output
wire [47:0] DSP_PCOUT;                               // 48-bit output: Cascade output

// Control: 1-bit (each) output: Control Inputs/Status Bits
wire        DSP_OVERFLOW;                            // 1-bit output: Overflow in add/acc output
wire        DSP_PATTERNBDETECT;                      // 1-bit output: Pattern bar detect output
wire        DSP_PATTERNDETECT;                       // 1-bit output: Pattern detect output
wire        DSP_UNDERFLOW;                           // 1-bit output: Underflow in add/acc output

// Data: 4-bit (each) output: Data Ports
wire  [3:0] DSP_CARRYOUT;                            // 4-bit output: Carry output
wire [47:0] DSP_P;                                   // 48-bit output: Primary data output

// Cascade: 30-bit (each) input: Cascade Ports
wire [29:0] DSP_ACIN            = 0;                 // 30-bit input: A cascade data input
wire [17:0] DSP_BCIN            = 0;                 // 18-bit input: B cascade input
wire        DSP_CARRYCASCIN     = 0;                 // 1-bit input: Cascade carry input
wire        DSP_MULTSIGNIN      = 0;                 // 1-bit input: Multiplier sign input
wire [47:0] DSP_PCIN            = 0;                 // 48-bit input: P cascade input

// Control: 4-bit (each) input: Control Inputs/Status Bits
wire  [3:0] DSP_ALUMODE         = 4'b     0000;      // 4-bit input: ALU control input
wire  [2:0] DSP_CARRYINSEL      = 3'b      000;      // 3-bit input: Carry select input
wire        DSP_CEINMODE        = 1;                 // 1-bit input: Clock enable input for INMODEREG
wire        DSP_CLK             = clk;               // 1-bit input: Clock input
wire  [4:0] DSP_INMODE          = 5'b   0_0000;      // 5-bit input: INMODE control input
wire  [6:0] DSP_OPMODE          = 7'b 000_0000;      // 7-bit input: Operation mode input
wire        DSP_RSTINMODE       = 0;                 // 1-bit input: Reset input for INMODEREG

// Data: 30-bit (each) input: Data Ports
wire [29:0] DSP_A               = 0;                 // 30-bit input: A data input
wire [17:0] DSP_B               = 0;                 // 18-bit input: B data input
wire [47:0] DSP_C               = 0;                 // 48-bit input: C data input
wire        DSP_CARRYIN         = 0;                 // 1-bit input: Carry input signal
wire [24:0] DSP_D               = 0;                 // 25-bit input: D data input

// Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs
wire        DSP_CEA1            = 1;                 // 1-bit input: Clock enable input for 1st stage AREG
wire        DSP_CEA2            = 1;                 // 1-bit input: Clock enable input for 2nd stage AREG
wire        DSP_CEAD            = 1;                 // 1-bit input: Clock enable input for ADREG
wire        DSP_CEALUMODE       = 1;                 // 1-bit input: Clock enable input for ALUMODERE
wire        DSP_CEB1            = 1;                 // 1-bit input: Clock enable input for 1st stage BREG
wire        DSP_CEB2            = 1;                 // 1-bit input: Clock enable input for 2nd stage BREG
wire        DSP_CEC             = 1;                 // 1-bit input: Clock enable input for CREG
wire        DSP_CECARRYIN       = 1;                 // 1-bit input: Clock enable input for CARRYINREG
wire        DSP_CECTRL          = 1;                 // 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG
wire        DSP_CED             = 1;                 // 1-bit input: Clock enable input for DREG
wire        DSP_CEM             = 1;                 // 1-bit input: Clock enable input for MREG
wire        DSP_CEP             = 1;                 // 1-bit input: Clock enable input for PREG
wire        DSP_RSTA            = 0;                 // 1-bit input: Reset input for AREG
wire        DSP_RSTALLCARRYIN   = 0;                 // 1-bit input: Reset input for CARRYINREG
wire        DSP_RSTALUMODE      = 0;                 // 1-bit input: Reset input for ALUMODEREG
wire        DSP_RSTB            = 0;                 // 1-bit input: Reset input for BREG
wire        DSP_RSTC            = 0;                 // 1-bit input: Reset input for CREG
wire        DSP_RSTCTRL         = 0;                 // 1-bit input: Reset input for OPMODEREG and CARRYINSELREG
wire        DSP_RSTD            = 0;                 // 1-bit input: Reset input for DREG and ADREG
wire        DSP_RSTM            = 0;                 // 1-bit input: Reset input for MREG
wire        DSP_RSTP            = 0;                 // 1-bit input: Reset input for PREG

DSP48E1 #(
	// Feature Control Attributes: Data Path Selection
	.A_INPUT("DIRECT"),               // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
	.B_INPUT("DIRECT"),               // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
	.USE_DPORT("FALSE"),              // Select D port usage (TRUE or FALSE)
	.USE_MULT("NONE"),                // Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")
	// Pattern Detector Attributes: Pattern Detection Configuration
	.AUTORESET_PATDET("NO_RESET"),    // "NO_RESET", "RESET_MATCH", "RESET_NOT_MATCH" 
	.MASK(48'h3fffffffffff),          // 48-bit mask value for pattern detect (1=ignore)
	.PATTERN(48'h000000000000),       // 48-bit pattern match for pattern detect
	.SEL_MASK("MASK"),                // "C", "MASK", "ROUNDING_MODE1", "ROUNDING_MODE2" 
	.SEL_PATTERN("PATTERN"),          // Select pattern value ("PATTERN" or "C")
	.USE_PATTERN_DETECT("NO_PATDET"), // Enable pattern detect ("PATDET" or "NO_PATDET")
	// Register Control Attributes: Pipeline Register Configuration
	.ACASCREG(1),                     // Number of pipeline stages between A/ACIN and ACOUT (0, 1 or 2)
	.ADREG(1),                        // Number of pipeline stages for pre-adder (0 or 1)
	.ALUMODEREG(0),                   // Number of pipeline stages for ALUMODE (0 or 1)
	.AREG(1),                         // Number of pipeline stages for A (0, 1 or 2)
	.BCASCREG(1),                     // Number of pipeline stages between B/BCIN and BCOUT (0, 1 or 2)
	.BREG(1),                         // Number of pipeline stages for B (0, 1 or 2)
	.CARRYINREG(1),                   // Number of pipeline stages for CARRYIN (0 or 1)
	.CARRYINSELREG(1),                // Number of pipeline stages for CARRYINSEL (0 or 1)
	.CREG(1),                         // Number of pipeline stages for C (0 or 1)
	.DREG(1),                         // Number of pipeline stages for D (0 or 1)
	.INMODEREG(1),                    // Number of pipeline stages for INMODE (0 or 1)
	.MREG(1),                         // Number of multiplier pipeline stages (0 or 1)
	.OPMODEREG(0),                    // Number of pipeline stages for OPMODE (0 or 1)
	.PREG(1),                         // Number of pipeline stages for P (0 or 1)
	.USE_SIMD("ONE48")                // SIMD selection ("ONE48", "TWO24", "FOUR12")
) DSP_DSP48E1 (
	// Cascade: 30-bit (each) output: Cascade Ports
	.ACOUT(DSP_ACOUT),                   // 30-bit output: A port cascade output
	.BCOUT(DSP_BCOUT),                   // 18-bit output: B port cascade output
	.CARRYCASCOUT(DSP_CARRYCASCOUT),     // 1-bit output: Cascade carry output
	.MULTSIGNOUT(DSP_MULTSIGNOUT),       // 1-bit output: Multiplier sign cascade output
	.PCOUT(DSP_PCOUT),                   // 48-bit output: Cascade output
	// Control: 1-bit (each) output: Control Inputs/Status Bits
	.OVERFLOW(DSP_OVERFLOW),             // 1-bit output: Overflow in add/acc output
	.PATTERNBDETECT(DSP_PATTERNBDETECT), // 1-bit output: Pattern bar detect output
	.PATTERNDETECT(DSP_PATTERNDETECT),   // 1-bit output: Pattern detect output
	.UNDERFLOW(DSP_UNDERFLOW),           // 1-bit output: Underflow in add/acc output
	// Data: 4-bit (each) output: Data Ports
	.CARRYOUT(DSP_CARRYOUT),             // 4-bit output: Carry output
	.P(dsp_p),                           // 48-bit output: Primary data output
	// Cascade: 30-bit (each) input: Cascade Ports
	.ACIN(DSP_ACIN),                     // 30-bit input: A cascade data input
	.BCIN(DSP_BCIN),                     // 18-bit input: B cascade input
	.CARRYCASCIN(DSP_CARRYCASCIN),       // 1-bit input: Cascade carry input
	.MULTSIGNIN(DSP_MULTSIGNIN),         // 1-bit input: Multiplier sign input
	.PCIN(DSP_PCIN),                     // 48-bit input: P cascade input
	// Control: 4-bit (each) input: Control Inputs/Status Bits
	.ALUMODE(dsp_alumode),               // 4-bit input: ALU control input
	.CARRYINSEL(DSP_CARRYINSEL),         // 3-bit input: Carry select input
	.CEINMODE(DSP_CEINMODE),             // 1-bit input: Clock enable input for INMODEREG
	.CLK(DSP_CLK),                       // 1-bit input: Clock input
	.INMODE(DSP_INMODE),                 // 5-bit input: INMODE control input
	.OPMODE(dsp_opmode),                 // 7-bit input: Operation mode input
	.RSTINMODE(DSP_RSTINMODE),           // 1-bit input: Reset input for INMODEREG
	// Data: 30-bit (each) input: Data Ports
	.A(dsp_ab[47:18]),                   // 30-bit input: A data input
	.B(dsp_ab[17:0]),                    // 18-bit input: B data input
	.C(dsp_c),                           // 48-bit input: C data input
	.CARRYIN(DSP_CARRYIN),               // 1-bit input: Carry input signal
	.D(DSP_D),                           // 25-bit input: D data input
	// Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs
	.CEA1(DSP_CEA1),                     // 1-bit input: Clock enable input for 1st stage AREG
	.CEA2(DSP_CEA2),                     // 1-bit input: Clock enable input for 2nd stage AREG
	.CEAD(DSP_CEAD),                     // 1-bit input: Clock enable input for ADREG
	.CEALUMODE(DSP_CEALUMODE),           // 1-bit input: Clock enable input for ALUMODERE
	.CEB1(DSP_CEB1),                     // 1-bit input: Clock enable input for 1st stage BREG
	.CEB2(DSP_CEB2),                     // 1-bit input: Clock enable input for 2nd stage BREG
	.CEC(DSP_CEC),                       // 1-bit input: Clock enable input for CREG
	.CECARRYIN(DSP_CECARRYIN),           // 1-bit input: Clock enable input for CARRYINREG
	.CECTRL(DSP_CECTRL),                 // 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG
	.CED(DSP_CED),                       // 1-bit input: Clock enable input for DREG
	.CEM(DSP_CEM),                       // 1-bit input: Clock enable input for MREG
	.CEP(DSP_CEP),                       // 1-bit input: Clock enable input for PREG
	.RSTA(DSP_RSTA),                     // 1-bit input: Reset input for AREG
	.RSTALLCARRYIN(DSP_RSTALLCARRYIN),   // 1-bit input: Reset input for CARRYINREG
	.RSTALUMODE(DSP_RSTALUMODE),         // 1-bit input: Reset input for ALUMODEREG
	.RSTB(DSP_RSTB),                     // 1-bit input: Reset input for BREG
	.RSTC(DSP_RSTC),                     // 1-bit input: Reset input for CREG
	.RSTCTRL(DSP_RSTCTRL),               // 1-bit input: Reset input for OPMODEREG and CARRYINSELREG
	.RSTD(DSP_RSTD),                     // 1-bit input: Reset input for DREG and ADREG
	.RSTM(DSP_RSTM),                     // 1-bit input: Reset input for MREG
	.RSTP(DSP_RSTP)                      // 1-bit input: Reset input for PREG
);

/********************************* END TEMPLATE: xilinx_dsp48e1.v *********************************/

endmodule

