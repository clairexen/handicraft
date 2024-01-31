#!/bin/bash

ii=$1
echo "Creating files for ex_${ii}.."

rm -rf binfiles/ex_${ii}.*
rm -rf tmpfiles/ex_${ii}.*

file_v=tmpfiles/ex_${ii}.v
file_pcf=tmpfiles/ex_${ii}.pcf

echo "// $file_v" > $file_v
echo "# $file_pcf" > $file_pcf
rm -f binfiles/ex_${ii}.*

num_clks=$(( RANDOM % 7 + 1 ))
num_ens=$(( 8 - $num_clks ))

printf "module top(input [%d:0] clk, input [%d:0] en, input x, output y);\n" $((num_clks-1)) $((num_ens-1)) >> $file_v
last_out_wire="x"

for x in {1..2} {4..9} {11..12}; do
for y in {1..16}; do
	clb_pack_list=""
	ff_type=$(( RANDOM % 3 ))
	ff_clkidx=$(( RANDOM % $num_clks ))
	ff_enidx=$(( RANDOM % $num_ens ))
	for z in {0..7}; do
		val=$(head -c128 /dev/urandom | md5sum | cut -c1-4)
		{
			used_input=$(( RANDOM % 4 ))
			printf "  wire alice_x%02d_y%02d_z%d_out;\n" $x $y $z
			printf "  wire alice_x%02d_y%02d_z%d_out_q;\n" $x $y $z
			printf "  (* syn_noprune *)\n"
			printf "  SB_LUT4 #(\n"
			printf "    .LUT_INIT(16'h%s)\n" $val
			printf "  ) alice_x%02d_y%02d_z%d (\n" $x $y $z
			for k in {0..3}; do
				if [ $k -eq $used_input ]; then
					printf "    .I%d(%s),\n" $k $last_out_wire
				else
					printf "    .I%d(1'b0),\n" $k
				fi
			done
			printf "    .O(alice_x%02d_y%02d_z%d_out)\n" $x $y $z
			printf "  );\n"
			case $ff_type in
				0)
					printf "  assign alice_x%02d_y%02d_z%d_out_q = alice_x%02d_y%02d_z%d_out;\n" $x $y $z $x $y $z
					;;
				1)
					printf "  SB_DFF alice_x%02d_y%02d_z%d_dff (\n" $x $y $z
					printf "    .C(clk[%d]),\n" $ff_clkidx
					printf "    .D(alice_x%02d_y%02d_z%d_out),\n" $x $y $z
					printf "    .Q(alice_x%02d_y%02d_z%d_out_q)\n" $x $y $z
					printf "  );\n"
					;;
				2)
					printf "  SB_DFFE alice_x%02d_y%02d_z%d_dff (\n" $x $y $z
					printf "    .C(clk[%d]),\n" $ff_clkidx
					printf "    .E(en[%d]),\n" $ff_enidx
					printf "    .D(alice_x%02d_y%02d_z%d_out),\n" $x $y $z
					printf "    .Q(alice_x%02d_y%02d_z%d_out_q)\n" $x $y $z
					printf "  );\n"
					;;
			esac
		} >> $file_v
		case $ff_type in
			0) printf "ble_pack alice_x%02d_y%02d_z%d_ble {alice_x%02d_y%02d_z%d}\n" $x $y $z $x $y $z >> $file_pcf ;;
			*) printf "ble_pack alice_x%02d_y%02d_z%d_ble {alice_x%02d_y%02d_z%d,alice_x%02d_y%02d_z%d_dff}\n" $x $y $z $x $y $z $x $y $z >> $file_pcf ;;
		esac
		last_out_wire=$(printf "alice_x%02d_y%02d_z%d_out_q" $x $y $z)
		[ -n "$clb_pack_list" ] && clb_pack_list="$clb_pack_list,"
		clb_pack_list="$clb_pack_list$(printf "alice_x%02d_y%02d_z%d_ble" $x $y $z)"
	done
	printf "clb_pack alice_x%02d_y%02d_clb {%s}\n" $x $y $clb_pack_list >> $file_pcf
	printf "set_location alice_x%02d_y%02d_clb %d %d\n" $x $y $x $y >> $file_pcf
	printf "#dffinfo %02d %02d %d %d %d\n" $x $y $ff_type $ff_clkidx $ff_enidx >> $file_pcf
done; done

printf "  assign y = %s;\n" $last_out_wire >> $file_v
printf "endmodule\n" >> $file_v

