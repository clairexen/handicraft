#include <ap_fixed.h>

void hlsbugtst4(ap_fixed<128, 64> b1, ap_fixed<128, 64> b2, ap_fixed<128, 64> b3, ap_fixed<128, 64> &y)
{
	y = (b1 - b2) * b3;
}

