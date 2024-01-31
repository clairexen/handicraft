#include <stdio.h>
#include <assert.h>
#include <rvintrin.h>

static inline long rdinstret() { int64_t rd; asm volatile ("rdinstret %0" : "=r"(rd) : : "memory"); return rd; }

uint64_t testdata[] = {
	0x23ebe88c0224f4f4, 0xfc0b608216e98735, 0x1181c1834012245d, 0x785120e3505ed169,
	0x348bd3ce10ec00ec, 0xc29d31ec97cd5839, 0x40b134ab8a3dee5d, 0xd9760a7805fd495c,
	0x45a62d3d5d3e9462, 0x50904697486591bc, 0x41d9737bc75ae8fa, 0x57b7a84a44fba01c,
	0x971658bc2f5bdee5, 0x660844a83b5bf0a2, 0xa4e75609afe6f1e7, 0xeea6d21a14e2f2a6,
	0xb6c7b766a7283579, 0xeecbe690a8ca9b3f, 0xf47bc36040d5c7db, 0x08e4b3a457dcfbb2,
	0xeb84c177ddcbc044, 0x2a29a1a670146f0e, 0x77a319564621b96f, 0xa0656e24c67960ef,
	0xc4d3fa1cbed3f838, 0xda59f45b4ff136bd, 0x4d46f0aea5c51215, 0xc187302f62ddff8f,
	0x9bcf97e7bee9ecea, 0xeb48d975876dbc02, 0xe5be1e77d5548b1b, 0xd97df94820280bd1,
	0x77d702fe375ead27, 0xd09aed411c287e67, 0xfad3475d201b6552, 0x4fe5172d0b3233eb,
	0xeaac9c3082e06d96, 0xebf906b59100ee69, 0xd3bd2f3362e2a2d7, 0xe9e11e22d4ff72ec,
	0xa87fd17c7aa67951, 0x4d2c3d368a14c0e1, 0xf64b8fbfccca3b70, 0x1f19cb2d5f163944,
	0xd0284d48bae75d56, 0xa8d372ea361f4876, 0x57de7708dffc2910, 0xf09dc7f72136d60c,
	0x0a1250e6aa45b0fe, 0x5e9233f7f178efe0, 0x35f9cccb12c91dcc, 0x440ea77ac880b722,
	0xa1718657330caadc, 0xf71dc051e1a811d1, 0x80e45917a287267e, 0x558d3eac0764e3c0,
	0x0cb4ebb59ded77e8, 0x9ec26c545e69a6dd, 0x25e3c146f5f1ed25, 0x6a927ae3726116ad,
	0x64da78e9e6218fa4, 0x2f9acf5b21924cb6, 0x892086d7a233f049, 0x588a4eb770beb74f,
	0x9b9a1496446aa167, 0xa262031c282399d2, 0xcdf761780b293a81, 0xeec6712832ff8c8a,
	0x92cdbac0c3f92f28, 0xecab3264d04cc946, 0x98dca1f92ac3e9b2, 0xa9f39819fe0810a4,
	0x3b47c8aef7069053, 0x5b51168d2ba619c6, 0xe5c9f9d8eb60f1ae, 0xca7325ffc0ec13b3,
	0x5f277a5544771ff3, 0x06320a48dc8e7744, 0xb504c513757e84f2, 0xaf4beb5e347c4d4a,
	0xc183342d3ec368d6, 0xfc88fdfdb0e423a9, 0x2ab745d42eb7d227, 0x1ea167c7f9a069fb,
	0x9f78d394170e0962, 0x5b563d9c364432df, 0xc2a0c4430c2c089f, 0xc9017fe0c1e0075d,
	0x2d00e5b494acbe88, 0x28a06c9314145671, 0x06671ba6c776dd4d, 0x4901abe64704b929,
	0xce32bc28a7e7c825, 0xa936e86e185e3bb3, 0xd6ea69a5401ef7e5, 0xe953c0edaa16cd50,
	0xde4bda08b7c2478b, 0xa953445321254b23, 0x42c4c0beffc43e41, 0x65b7deab4d09bfcb,
	0xd990cdbf1e42e343, 0x3961a5fc8c9818b9, 0xfc5867e489a84c59, 0xb783784936aa10ac,
	0x4f4f02a44a3a52c3, 0xffe82c262b6f1639, 0xcd82f4c975791c1a, 0xdf0a9823829815e7,
	0x58f4dfb38dd40151, 0x6cf998dc7ce3c55b, 0x87d8b8327fdb6ecf, 0x640a7f325d341af9,
	0x034efd83950f640c, 0x95baac0e1a079ec9, 0xed2d1d0a7f68e6a0, 0xf60a2b44303924c0,
	0x780def655779f7b6, 0x76d829811faede52, 0x5689ef58d3707c4a, 0x63d5134bc59ce774,
	0xd7adc2e0fb18f1ea, 0x0f849156de47dd8b, 0x426742a911ebe49f, 0x030c0545afe7d580,
	0x392ab9be505a7e3d, 0xc1248b3cc84e705b, 0x4fe18ad081b9629d, 0xb4b58cee3e2fab11,
	0x38281f4e7fff9544, 0x22afd7ca02698da0, 0xc6ec9d8b609ea56e, 0x6fb66193378f1ac4,
	0x5196c22f7dd4ef30, 0x3bb66adba23965bb, 0xa16dee33f24f2d70, 0xb839d3996b830891,
	0xe33a1c012ef4874f, 0xeda5712546237682, 0x551cf8e41cdcec58, 0x71e41a4a8c6b6228,
	0x53aa91f3a2433c43, 0x875607871e817805, 0x8ca8a2fa387d606b, 0xa3b63e2f751113e5,
	0x461e935ff9255d95, 0x4bf968379405ac77, 0x2ff8ae180c44c3f6, 0x7847cdb53a0d2a7f,
	0xab31c1fb931ad4c1, 0x375f9d2a10ef67a3, 0x2dec6a0886c5cf14, 0x7646970852a047a8,
	0x830df39b6d9b7bf7, 0xf7b17ac4aee62904, 0x2520c5cdaa568912, 0xf35d917a8dcde1b3,
	0x019c08d61d394cbb, 0x104d51f430dc1b33, 0x192dd82f05bb8c48, 0xdb6e605a99d73f46,
	0x45137f1ed65e4d64, 0x51a4beaae3a61e12, 0x525bce0b14b34cde, 0x0799ea595ec226a5,
	0xd7c1cd792c305750, 0x0fdc1e623aaf9f8c, 0xe11ace541e36340e, 0xe30dbe9b129b2801,
	0x049f4df7acc97d8d, 0xc7585931ebb14e18, 0x07d9b161e4ae96d3, 0x021c98e786691a3f,
	0x5ed483c436c83744, 0xe97e85fec56fd2d8, 0xe23ea52ecd4b92ed, 0x962cf88afa5690ce,
	0xc11143764c8e9fe0, 0x1ca374e27e02f308, 0xcaebe8cfb8125054, 0x9b7d966ad2bd57aa,
	0xf3f810afd4ea4b27, 0xf1a4bb4bd8856577, 0xb596374f8c444733, 0x646c10fb26eb6307,
	0x59ba6796fcb116ec, 0xac7532793ee87070, 0xc1fb54a1116d6b18, 0x7aaaa6f345d25ce4,
	0xde29eb01a59ee700, 0xac02b532152493de, 0x7bc87b0b547a25fe, 0x2dbda9f8335539e5,
	0xb0a3a2c6574fd5ff, 0xb9a4a947b294d196, 0x42ee49b66f0d07a7, 0x3be69efc2e75f99a,
	0x9cd09f32caab4653, 0x3b3790af2433d7a0, 0xd668e9dbda7d1d73, 0x85cbb2b118313b12
};

// ---- BEGIN copy&paste from prince_ref.h ----
// https://github.com/sebastien-riou/prince-c-ref/blob/master/include/prince_ref.h

/*
Copyright 2016 Sebastien Riou
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * The 4 bit Prince sbox. Only the 4 lsb are takken into account.
 */
static unsigned int prince_ref_sbox(unsigned int nibble){
  const unsigned int sbox[] = {
    0xb, 0xf, 0x3, 0x2,
    0xa, 0xc, 0x9, 0x1,
    0x6, 0x7, 0x8, 0x0,
    0xe, 0x5, 0xd, 0x4
  };
  return sbox[nibble & 0xF];
}

/**
 * The 4 bit Prince inverse sbox. Only the 4 lsb are takken into account.
 */
static unsigned int prince_ref_sbox_inv(unsigned int nibble){
  const unsigned int sbox[] = {
    0xb, 0x7, 0x3, 0x2,
    0xf, 0xd, 0x8, 0x9,
    0xa, 0x6, 0x4, 0x0,
    0x5, 0xe, 0xc, 0x1
  };
  return sbox[nibble & 0xF];
}

// ---- END copy&paste from prince_ref.h ----


// ---- BEGIN prince s-box constants from SCARV project ----
// https://github.com/scarv/libscarv/blob/master/src/libscarv/block/prince/riscv-xcrypto/prince_opt.S

// Copyright (C) 2019 SCARV project <info@scarv.org>
//
// Use of this source code is restricted per the MIT license, a copy of which
// can be found at https://opensource.org/licenses/MIT (or should be included
// as LICENSE.txt within the associated archive or repository).

uint64_t sbox_lut = 0x4d5e087619ca23fb;
uint64_t inv_sbox_lut = 0x1ce5046a98df237b;

// ---- END prince s-box constants from SCARV project ----


uint64_t reference_prince_sbox(uint64_t arg) 
{
	uint64_t ret = 0;
	for (int i = 0; i < 64; i+=4)
		ret |= (uint64_t)prince_ref_sbox(arg >> i) << i;
	return ret;
}

uint64_t reference_prince_sbox_inv(uint64_t arg) 
{
	uint64_t ret = 0;
	for (int i = 0; i < 64; i+=4)
		ret |= (uint64_t)prince_ref_sbox_inv(arg >> i) << i;
	return ret;
}

uint64_t xcrypto_lut4(uint64_t data, uint64_t lut)
{
	uint64_t ret = 0;
	for (int i = 0; i < 64; i+=4)
		ret |= (15 & (lut >> (4*(15 & (data >> i))))) << i;
	return ret;
}

uint64_t xcrypto_prince_sbox(uint64_t arg)
{
	return xcrypto_lut4(arg, sbox_lut);
}

uint64_t xcrypto_prince_sbox_inv(uint64_t arg)
{
	return xcrypto_lut4(arg, inv_sbox_lut);
}

uint64_t bitmanip_prince_sbox(uint64_t arg)
{
	// TBD
	return xcrypto_lut4(arg, sbox_lut);
}

uint64_t bitmanip_prince_sbox_inv(uint64_t arg)
{
	// TBD
	return xcrypto_lut4(arg, inv_sbox_lut);
}

int main()
{
	for (int i = 0; i < sizeof(testdata)/sizeof(*testdata); i++)
	{
		long a = testdata[i];
		long b = reference_prince_sbox(a);
		long b_xcrypto = xcrypto_prince_sbox(a);
		long b_bitmanip = bitmanip_prince_sbox(a);
		long c = reference_prince_sbox_inv(b);
		long c_xcrypto = xcrypto_prince_sbox_inv(b);
		long c_bitmanip = bitmanip_prince_sbox_inv(b);

		printf("%3d: %016lx %016lx (%016lx, %016lx) %016lx (%016lx, %016lx)\n", i, a, b, b_xcrypto, b_bitmanip, c, c_xcrypto, c_bitmanip);

		assert(a == c);
		assert(b == b_xcrypto);
		assert(c == c_xcrypto);
		assert(b == b_bitmanip);
		assert(c == c_bitmanip);
	}
	
	return 0;
}

