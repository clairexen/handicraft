#!/bin/bash

iverilog -o array_multiplier_tb array_multiplier.v array_multiplier_tb.v
iverilog -o array_multiplier_pipeline_tb array_multiplier_pipeline.v array_multiplier_pipeline_tb.v
paste <( ./array_multiplier_tb ) <( ./array_multiplier_pipeline_tb ) | expand -t60

