#!/bin/bash

set -ex
. ../settings.sh

rm -f vivado.log
vivado -mode batch -source playground.tcl -nojournal

for prefix in data_*_A.bit; do
	prefix=${prefix%_A.bit}
	../tools/bitread -xzo ${prefix}_A.asc < ${prefix}_A.bit
	../tools/bitread -xzo ${prefix}_B.asc < ${prefix}_B.bit
	../tools/bitdiff -d ${prefix}_A.asc ${prefix}_B.asc > ${prefix}.dat
done

{
	echo "| Frame ID | Frame Top | Frame Row | Frame Col | Frame Minor | Region | Grid Col | Grid Row | Tile |"
	echo "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
	while read fid fid_top fid_row fid_col fid_minor region grid_col grid_row tile_type tile_lr tile_pos; do
		echo "| $fid | $fid_top | $fid_row | $fid_col | $fid_minor | $region | $grid_col | $grid_row | ${tile_type}_${tile_lr}_${tile_pos} |"
	done < <( cat data_*.dat | sort | tr '_.,[]' ' ' | tr -s ' ' | cut -f1,3-6,9-14 -d' ' | uniq; )
} > table.md

