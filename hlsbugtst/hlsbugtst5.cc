#include <ap_int.h>

const ap_uint<7> prime_table[32] = {
	0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
	59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127
};

ap_uint<7> hlsbugtst5(ap_uint<5> index) {
	return prime_table[index];
}
