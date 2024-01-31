#include <ap_fixed.h>

extern void hlsbugtst4(ap_fixed<128, 64> b1, ap_fixed<128, 64> b2, ap_fixed<128, 64> b3, ap_fixed<128, 64> &b);

int main()
{
	ap_fixed< 64, 32> a1 = 62582.4, a2 = 153252, a3 = 2.72708e-06, a = (a1 - a2) * a3;
	ap_fixed<128, 64> b1 = 62582.4, b2 = 153252, b3 = 2.72708e-06, b = (b1 - b2) * b3;

	std::cout << "Test ap_fixed< 64, 32>: (" << a1 << " - " << a2 << ") * " << a3 << " = " << a << std::endl;
	std::cout << "Test ap_fixed<128, 64>: (" << b1 << " - " << b2 << ") * " << b3 << " = " << b << std::endl;

	ap_fixed<128, 64> y;
	hlsbugtst4(b1, b2, b3, y);

	std::cout << "HLS function output for ap_fixed<128, 64>: " << y << std::endl;

	if (y != b) {
		std::cout << "Error: mismatch between HLS function and C model!" << std::endl;
		return 1;
	}

	return 0;
}

