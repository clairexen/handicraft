
module example ( clk, rst, ctrl, step, out );
  input [3:0] step;
  output [3:0] out;
  input clk, rst, ctrl;
  wire   n65, n66, n67, N10, n8, n9, n10, n11, n12, n13, n14, n16, n17, n18,
         n19, n20, n24, n25, n26, n27, n28, n29, n30, n32, n33, n34, n35, n36,
         n37, n38, n39, n40, n41, n42, n43, n44, n45, n46, n47, n48, n49, n50,
         n51, n52, n56, n58, n59, n60, n61, n62, n63, n64;
  wire   [2:0] state;

  DF1 \state_reg[0]  ( .D(n8), .C(clk), .Q(state[0]), .QN(n11) );
  DF1 \state_reg[1]  ( .D(n51), .C(clk), .QN(n10) );
  DF1 \state_reg[2]  ( .D(n52), .C(clk), .Q(state[2]) );
  XNR31 U4 ( .A(step[3]), .B(n12), .C(n26), .Q(n24) );
  OAI212 U5 ( .A(n27), .B(n18), .C(n13), .Q(n26) );
  OAI222 U10 ( .A(n32), .B(n18), .C(ctrl), .D(n29), .Q(n34) );
  INV4 U32 ( .A(rst), .Q(N10) );
  DF3 \out_reg[2]  ( .D(n48), .C(clk), .Q(n66), .QN(n14) );
  DF3 \out_reg[0]  ( .D(n50), .C(clk), .Q(n67), .QN(n17) );
  DF3 \out_reg[3]  ( .D(n47), .C(clk), .Q(n65), .QN(n12) );
  DF1 \out_reg[1]  ( .D(n49), .C(clk), .Q(n59), .QN(n16) );
  INV15 U40 ( .A(n17), .Q(out[0]) );
  XOR21 U41 ( .A(n58), .B(n35), .Q(n29) );
  XNR21 U42 ( .A(n33), .B(n35), .Q(n32) );
  INV3 U43 ( .A(n56), .Q(n36) );
  AOI2111 U44 ( .A(step[2]), .B(n29), .C(ctrl), .D(n30), .Q(n28) );
  INV3 U45 ( .A(n61), .Q(n30) );
  NAND22 U46 ( .A(n58), .B(n14), .Q(n61) );
  AOI221 U47 ( .A(n32), .B(n19), .C(n33), .D(n14), .Q(n27) );
  INV3 U48 ( .A(n28), .Q(n13) );
  INV3 U49 ( .A(n62), .Q(n33) );
  INV3 U50 ( .A(step[1]), .Q(n64) );
  INV3 U51 ( .A(n38), .Q(n63) );
  NAND22 U52 ( .A(step[0]), .B(n17), .Q(n56) );
  NOR21 U53 ( .A(n17), .B(n20), .Q(n38) );
  XNR21 U54 ( .A(n19), .B(n66), .Q(n35) );
  INV3 U55 ( .A(step[2]), .Q(n19) );
  NOR21 U56 ( .A(n10), .B(ctrl), .Q(n44) );
  INV3 U57 ( .A(n46), .Q(n9) );
  NOR31 U58 ( .A(rst), .B(state[2]), .C(n44), .Q(n46) );
  NOR31 U59 ( .A(n11), .B(rst), .C(n43), .Q(n51) );
  NOR21 U60 ( .A(n9), .B(ctrl), .Q(n43) );
  INV3 U61 ( .A(n45), .Q(n8) );
  AOI211 U62 ( .A(state[0]), .B(n43), .C(n9), .Q(n45) );
  NOR31 U63 ( .A(n10), .B(rst), .C(n44), .Q(n52) );
  NAND22 U64 ( .A(state[2]), .B(N10), .Q(n25) );
  INV3 U65 ( .A(ctrl), .Q(n18) );
  INV15 U66 ( .A(n16), .Q(out[1]) );
  INV15 U67 ( .A(n12), .Q(out[3]) );
  INV3 U68 ( .A(step[0]), .Q(n20) );
  AOI210 U69 ( .A(n67), .B(n20), .C(n36), .Q(n42) );
  OAI220 U70 ( .A(n9), .B(n17), .C(n42), .D(n25), .Q(n50) );
  INV15 U71 ( .A(n14), .Q(out[2]) );
  XOR21 U72 ( .A(step[1]), .B(n16), .Q(n41) );
  OAI220 U73 ( .A(n9), .B(n16), .C(n25), .D(n40), .Q(n49) );
  INV3 U74 ( .A(n37), .Q(n60) );
  OAI220 U75 ( .A(n39), .B(n18), .C(ctrl), .D(n37), .Q(n40) );
  XNR20 U76 ( .A(n38), .B(n41), .Q(n39) );
  XNR21 U77 ( .A(n36), .B(n41), .Q(n37) );
  OAI222 U78 ( .A(n59), .B(n56), .C(n60), .D(n64), .Q(n58) );
  OAI222 U79 ( .A(n16), .B(n63), .C(n64), .D(n39), .Q(n62) );
  OAI221 U80 ( .A(n9), .B(n12), .C(n24), .D(n25), .Q(n47) );
  OAI221 U81 ( .A(n9), .B(n14), .C(n25), .D(n34), .Q(n48) );
endmodule

