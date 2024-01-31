#!/bin/bash

mkdir -p binfiles
mkdir -p tmpfiles

printf "\nall: all_targets\n\n" > tmpfiles/makefile
make_targets=""

for i in {0..99}; do
	ii=$(printf "%02d" $i)
	echo "Creating files for ex_${ii}.."

	rm -rf binfiles/ex_${ii}.*
	rm -rf tmpfiles/ex_${ii}.*

	file_v=tmpfiles/ex_${ii}.v
	file_pcf=tmpfiles/ex_${ii}.pcf
	file_bin=binfiles/ex_${ii}.bin
	file_val=binfiles/ex_${ii}.val

	echo "// $file_v" > $file_v
	echo "# $file_pcf" > $file_pcf
	echo -n > $file_val

	cat >> $file_v <<- EOT
		module top (
		  input clk,
		  input [7:0] sel,
		  input [15:0] wdata,
		  output [15:0] rdata,
		  input [7:0] addr
		);
	EOT

	idx=0
	rdataexpr=""

	for x in 03 10; do
	for y in 01 03 05 07 09 11 13 15; do
		cat >> $file_v <<- EOT
		  wire [15:0] rdata_$x$y;
		  SB_RAM256x16 ram_$x$y (
		    .RDATA(rdata_$x$y),
		    .RADDR(addr),
		    .RCLK(clk),
		    .RCLKE(1'b1),
		    .RE(1'b1),
		    .WADDR(addr),
		    .WCLK(clk),
		    .WCLKE(1'b1),
		    .WDATA(wdata),
		    .WE(sel == $idx),
		    .MASK(16'b0)
		  );
		EOT
		for z in 0 1 2 3 4 5 6 7 8 9 A B C D E F; do
			v0=$(head -c128 /dev/urandom | md5sum | head -c8)
			v1=$(head -c128 /dev/urandom | md5sum | head -c8)
			v2=$(head -c128 /dev/urandom | md5sum | head -c8)
			v3=$(head -c128 /dev/urandom | md5sum | head -c8)
			v4=$(head -c128 /dev/urandom | md5sum | head -c8)
			v5=$(head -c128 /dev/urandom | md5sum | head -c8)
			v6=$(head -c128 /dev/urandom | md5sum | head -c8)
			v7=$(head -c128 /dev/urandom | md5sum | head -c8)
			printf "ram_%s_%s.init_%s0 32 0x%s\n" $x $y $z $v0 >> $file_val
			printf "ram_%s_%s.init_%s1 32 0x%s\n" $x $y $z $v1 >> $file_val
			printf "ram_%s_%s.init_%s2 32 0x%s\n" $x $y $z $v2 >> $file_val
			printf "ram_%s_%s.init_%s3 32 0x%s\n" $x $y $z $v3 >> $file_val
			printf "ram_%s_%s.init_%s4 32 0x%s\n" $x $y $z $v4 >> $file_val
			printf "ram_%s_%s.init_%s5 32 0x%s\n" $x $y $z $v5 >> $file_val
			printf "ram_%s_%s.init_%s6 32 0x%s\n" $x $y $z $v6 >> $file_val
			printf "ram_%s_%s.init_%s7 32 0x%s\n" $x $y $z $v7 >> $file_val
			echo "defparam ram_$x$y.INIT_$z = 256'h$v7$v6$v5$v4$v3$v2$v1$v0;" >> $file_v
		done
		echo "set_location ram_$x$y ${x#0} ${y#0}" >> $file_pcf
		rdataexpr="$rdataexpr sel == $idx ? rdata_$x$y :"
		(( idx++ ))
	done; done

	echo "assign rdata =$rdataexpr 0;" >> $file_v
	echo "endmodule" >> $file_v

	printf "binfiles/ex_${ii}.bin:\n\ticecuberun tmpfiles/ex_${ii} &> tmpfiles/ex_${ii}.log\n\tmv tmpfiles/ex_${ii}.bin binfiles/\n\n" >> tmpfiles/makefile
	make_targets="$make_targets binfiles/ex_${ii}.bin"
done

printf "all_targets:$make_targets\n\n" >> tmpfiles/makefile
make -j4 -f tmpfiles/makefile

