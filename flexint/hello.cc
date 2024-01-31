#include "flexint.h"

int main()
{
	auto v1 = flexint<32, -8>::from_int(123);
	auto v2 = flexint<16, 4>::from_float(456789);
	auto v3 = flexint<32, 0>::mul(v1, v2);
}

