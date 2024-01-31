`timescale 1 ns / 1 ps
`default_nettype none

module testbench;
	reg ap_clk = 0;
	always #5 ap_clk = ~ap_clk;

	reg ap_rst = 1;
	initial begin
		repeat (100) @(posedge ap_clk);
		ap_rst <= 0;
	end

	reg ap_start = 1;

	wire ap_done;
	wire ap_idle;
	wire ap_ready;

	reg  [7:0]  pde_data_in_V_cnt_ls_V_dout;
	reg         pde_data_in_V_cnt_ls_V_empty_n = 0;
	wire        pde_data_in_V_cnt_ls_V_read;
	reg  [15:0] pde_data_in_V_pos_V_dout;
	reg         pde_data_in_V_pos_V_empty_n = 0;
	wire        pde_data_in_V_pos_V_read;

	reg  [7:0]  pdae_data_in_V_cnt_ls_V_dout;
	reg         pdae_data_in_V_cnt_ls_V_empty_n = 0;
	wire        pdae_data_in_V_cnt_ls_V_read;
	reg  [15:0] pdae_data_in_V_pos_V_dout;
	reg         pdae_data_in_V_pos_V_empty_n = 0;
	wire        pdae_data_in_V_pos_V_read;

	wire [7:0]  racalc_jobs_V_cnt_ls_V_din;
	reg         racalc_jobs_V_cnt_ls_V_full_n = 0;
	wire        racalc_jobs_V_cnt_ls_V_write;

	hls_uut uut (
		.ap_clk                            (ap_clk                           ),
		.ap_rst                            (ap_rst                           ),
		.ap_start                          (ap_start                         ),
		.ap_done                           (ap_done                          ),
		.ap_idle                           (ap_idle                          ),
		.ap_ready                          (ap_ready                         ),

		.pde_data_in_V_cnt_ls_V_dout       (pde_data_in_V_cnt_ls_V_dout      ),
		.pde_data_in_V_cnt_ls_V_empty_n    (pde_data_in_V_cnt_ls_V_empty_n   ),
		.pde_data_in_V_cnt_ls_V_read       (pde_data_in_V_cnt_ls_V_read      ),
		.pde_data_in_V_pos_V_dout          (pde_data_in_V_pos_V_dout         ),
		.pde_data_in_V_pos_V_empty_n       (pde_data_in_V_pos_V_empty_n      ),
		.pde_data_in_V_pos_V_read          (pde_data_in_V_pos_V_read         ),

		.pdae_data_in_V_cnt_ls_V_dout      (pdae_data_in_V_cnt_ls_V_dout     ),
		.pdae_data_in_V_cnt_ls_V_empty_n   (pdae_data_in_V_cnt_ls_V_empty_n  ),
		.pdae_data_in_V_cnt_ls_V_read      (pdae_data_in_V_cnt_ls_V_read     ),
		.pdae_data_in_V_pos_V_dout         (pdae_data_in_V_pos_V_dout        ),
		.pdae_data_in_V_pos_V_empty_n      (pdae_data_in_V_pos_V_empty_n     ),
		.pdae_data_in_V_pos_V_read         (pdae_data_in_V_pos_V_read        ),

		.racalc_jobs_V_cnt_ls_V_din        (racalc_jobs_V_cnt_ls_V_din       ),
		.racalc_jobs_V_cnt_ls_V_full_n     (racalc_jobs_V_cnt_ls_V_full_n    ),
		.racalc_jobs_V_cnt_ls_V_write      (racalc_jobs_V_cnt_ls_V_write     )
	);

	integer output_cnt = 0;
	`include "testdata.v"

	task pde_data(input [7:0] cnt_ls, input [15:0] pos);
		begin
			while (ap_rst) @(posedge ap_clk);

			pde_data_in_V_cnt_ls_V_dout <= cnt_ls;
			pde_data_in_V_pos_V_dout <= pos;

			pde_data_in_V_cnt_ls_V_empty_n <= 1;
			pde_data_in_V_pos_V_empty_n <= 1;
			@(posedge ap_clk);

			while ({pde_data_in_V_cnt_ls_V_empty_n, pde_data_in_V_pos_V_empty_n})
			begin
				if (pde_data_in_V_cnt_ls_V_read)    pde_data_in_V_cnt_ls_V_empty_n <= 0;
				if (pde_data_in_V_pos_V_read)       pde_data_in_V_pos_V_empty_n <= 0;
				@(posedge ap_clk);
			end
		end
	endtask

	task pdae_data(input [7:0] cnt_ls, input [15:0] pos);
		begin
			while (ap_rst) @(posedge ap_clk);

			pdae_data_in_V_cnt_ls_V_dout <= cnt_ls;
			pdae_data_in_V_pos_V_dout <= pos;

			pdae_data_in_V_cnt_ls_V_empty_n <= 1;
			pdae_data_in_V_pos_V_empty_n <= 1;
			@(posedge ap_clk);

			while ({pdae_data_in_V_cnt_ls_V_empty_n, pdae_data_in_V_pos_V_empty_n})
			begin
				if (pdae_data_in_V_cnt_ls_V_read)     pdae_data_in_V_cnt_ls_V_empty_n <= 0;
				if (pdae_data_in_V_pos_V_read)        pdae_data_in_V_pos_V_empty_n <= 0;
				@(posedge ap_clk);
			end
		end
	endtask

	task racalc_data(input [7:0] cnt_ls);
		reg [ 7:0] data_cnt_ls;
		begin
			while (ap_rst) @(posedge ap_clk);

			racalc_jobs_V_cnt_ls_V_full_n <= 1;
			@(posedge ap_clk);

			while (racalc_jobs_V_cnt_ls_V_full_n) begin
				if (racalc_jobs_V_cnt_ls_V_write) begin
					data_cnt_ls <= racalc_jobs_V_cnt_ls_V_din;
					racalc_jobs_V_cnt_ls_V_full_n <= 0;
				end
				@(posedge ap_clk);
			end

			$display("%s %02x=%02x", data_cnt_ls === cnt_ls ? "OK   " : "ERROR", data_cnt_ls, cnt_ls);

			output_cnt = output_cnt + 1;
			if (output_cnt > 100) $finish;
		end
	endtask

	initial begin
		testdata_pde;
	end

	initial begin
		testdata_pdae;
	end

	initial begin
		testdata_racalc;
	end
endmodule
