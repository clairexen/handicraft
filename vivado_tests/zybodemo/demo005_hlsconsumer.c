#include <stdbool.h>

bool hlsconsumer(unsigned char *in)
{
#pragma HLS INTERFACE ap_fifo port=in
#pragma HLS INTERFACE ap_ctrl_none register port=return
#pragma HLS RESOURCE variable=in core=AXI4Stream
	return *in;
}
