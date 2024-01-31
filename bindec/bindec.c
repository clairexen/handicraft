// This is a joke! Do not use this code to output binary values!
// See https://twitter.com/oe1cxw/status/801058586357022720 for context.

#include <stdio.h>
#include <assert.h>

int bindec_ref(unsigned char v)
{
	int r = 0; unsigned char t = 128;
	while (t) { r = 10*r + !!((v) & t); t >>= 1; }
	return r;
}

int bindec(unsigned char v)
{
	int r = v & 1;
	r += (~((v &   2) - 1)) &       10;
	r += (~((v &   4) - 1)) &      100;
	r += (~((v &   8) - 1)) &     1000;
	r += (~((v &  16) - 1)) &    10000;
	r += (~((v &  32) - 1)) &   100000;
	r += (~((v &  64) - 1)) &  1000000;
	r += (~((v & 128) - 1)) & 10000000;
	return r;
}

int main()
{
	int i = 0;
	for (i = 0; i < 256; i ++) {
		int a = bindec_ref(i), b = bindec(i);
		printf("%3d %08d %08d\n", i, a, b);
		assert(a == b);
	}
	return 0;
}

