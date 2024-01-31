module model(x, y, z, q);
input [7:0] x, y, z;
output q = (3*x + 4*y - z == 14) && (-2*x - 4*z <= -6) && (x - 3*y + z >= 15) && x < 10 && y < 10 && z < 10;
endmodule
