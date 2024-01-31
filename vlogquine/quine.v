module quine;
parameter t = {"integer i; initial begin $display(\"module quine;\");\n",
"$write(\"parameter t = {\\\"\"); for (i = 292*8-1; i > 0; i = i-8)\n",
"case (t[i-:8]) \"\\n\": $write(\"\\\\n\\\",\\n\\\"\"); \"\\\"\": $write(\"\\\\\\\"\");\n",
"\"\\\\\": $write(\"\\\\\\\\\"); default: $write(\"%s\", t[i-:8]); endcase\n",
"$display(\"\\\"};\"); $display(\"%s\", t); end endmodule"};
integer i; initial begin $display("module quine;");
$write("parameter t = {\""); for (i = 292*8-1; i > 0; i = i-8)
case (t[i-:8]) "\n": $write("\\n\",\n\""); "\"": $write("\\\"");
"\\": $write("\\\\"); default: $write("%s", t[i-:8]); endcase
$display("\"};"); $display("%s", t); end endmodule
