// http://stackoverflow.com/a/22535044/2213720

#include <stdio.h>
#include <stdint.h>

union {
    uint8_t opcode;
    struct {
        uint8_t z:3;
        uint8_t y:3;
        uint8_t x:2;
    };
    struct {
        uint8_t:3;
        uint8_t p:2;
        uint8_t q:1;
    };
} opcode;

int main()
{
	opcode.opcode = 0;
	opcode.p = 3;
	printf("%zd %d\n", sizeof(opcode), opcode.opcode);
	return 0;
}
