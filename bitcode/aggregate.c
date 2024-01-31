// Source: http://aggregate.org/MAGIC/
//
//
// @techreport{magicalgorithms,
// author={Henry Gordon Dietz},
// title={{The Aggregate Magic Algorithms}},
// institution={University of Kentucky},
// howpublished={Aggregate.Org online technical report},
// URL={http://aggregate.org/MAGIC/}
// }

unsigned int
reverse_a(register unsigned int x)
{
	x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
	x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
	x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
	x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
	return((x >> 16) | (x << 16));

}

unsigned int
reverse_b(register unsigned int x)
{
        register unsigned int y = 0x55555555;
        x = (((x >> 1) & y) | ((x & y) << 1));
        y = 0x33333333;
        x = (((x >> 2) & y) | ((x & y) << 2));
        y = 0x0f0f0f0f;
        x = (((x >> 4) & y) | ((x & y) << 4));
        y = 0x00ff00ff;
        x = (((x >> 8) & y) | ((x & y) << 8));
        return((x >> 16) | (x << 16));
}

unsigned int
g2b(unsigned int gray)
{
        gray ^= (gray >> 16);
        gray ^= (gray >> 8);
        gray ^= (gray >> 4);
        gray ^= (gray >> 2);
        gray ^= (gray >> 1);
        return(gray);
}

unsigned int
ones(register unsigned int x)
{
        /* 32-bit recursive reduction using SWAR...
	   but first step is mapping 2-bit values
	   into sum of 2 1-bit values in sneaky way
	*/
        x -= ((x >> 1) & 0x55555555);
        x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
        x = (((x >> 4) + x) & 0x0f0f0f0f);
        x += (x >> 8);
        x += (x >> 16);
        return(x & 0x0000003f);
}

unsigned int
lzc(register unsigned int x)
{
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return(32 - ones(x));
}

unsigned int
floor_log2(register unsigned int x)
{
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
	return(ones(x >> 1));
}

unsigned int
floor_log2_zero_undef(register unsigned int x)
{
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return(ones(x) - 1);
}

unsigned int
my_log2(register unsigned int x)
{
	register int y = (x & (x - 1));

	y |= -y;
	y >>= (32 - 1);
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
	return(ones(x >> 1) - y);
}

unsigned int
log2_zero_undef(register unsigned int x)
{
	register int y = (x & (x - 1));

	y |= -y;
	y >>= (32 - 1);
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return(ones(x) - 1 - y);
}

unsigned int
nlpo2(register unsigned int x)
{
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return(x+1);
}

unsigned int
msb32(register unsigned int x)
{
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return(x & ~(x >> 1));
}

unsigned int
tzc(register int x)
{
        return(ones((x & -x) - 1));
}

