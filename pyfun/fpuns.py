#!/usr/bin/env python3
#
# Experiments with floating point representations, comparing:
#   - Fpuns (newly proposed format by Claire Xenia Wolf)
#   - Posits (aka type-III Unums, as proposed by John L. Gustafson)
#   - IEEE floating point numbers


class GenericFloat:
    '''
    A generic arbitrary precision floating point object

    Attributes
    ----------
    exponent : int
        The exponent. Set to 0 for zero and to 1 for inf/nan.
    mantissa : int
        The (signed) mantissa, including the leading 1-bit.
        Set to 0 for zero and inf/nan.
    '''

    def __init__(self, exponent, mantissa):
        self.exponent = exponent
        self.mantissa = mantissa

    def __str__(self):
        if self.mantissa == 0:
            return "0" if self.exponent == 0 else "NaN"
        return bin(self.mantissa).replace("0b1", "0b1.") + f"e{exponent}"

class Fpun:
    '''
    A representation of an arbitrary precision Fpun

    From MSB to LSB bit:
    - Quadrant
      - sign-bit: 0 = positive, 1 = negative
      - supu-bit: 0 = magnitude smaller than one, 1 = magnitude bigger or equal one
      the supu bit is stored inverted when the sign bit is set (number is negative), and all
      futher bits are stored inverted if sign == supu.
    - Exponent 
      - 0b0                ( 1 bit) => 0
      - 0b10               ( 2 bit) => 1
      - 0b110a             ( 4 bit) => 0b1a
      - 0b1110ab           ( 6 bit) => 0b1ab
          ...
      - 0b111111110abcdefg (16 bit) => 0b1abcdefg
      i.e. the unary (zero-terminated) length L, followed by L-1 data bits if L>1. The decoded
      exponent is then either 0 for L=0, or an implicit 1-bit followed by the L-1 data bits.
    - Mantissa
      All remaining bits are mantissa bits (with the implicit leading 1-bit not stored in the encoded
      Fpun).

    Example Numbers:
     0b 00_000..000   Zero (exponent truncated -> round to zero)
     0b 00_000..001   For a 32-bit number we get L=29, so that's around 1.0e-160000000
     0b 01_000..000   One
     0b 01_111..111   For a 32-bit number we get L=29, so that's around 1.0e+160000000
     0b 10_000..000   Not-A-Number encoding
     0b 10_000..001   For a 32-bit number we get L=29, so that's around -1.0e+160000000
     0b 11_000..000   Negative One
     0b 11_111..111   For a 32-bit number we get L=29, so that's around -1.0e-160000000
    '''
    def __init__(self, value=None : GenericFloat):
        if value is None:
            self.sign = False
            self.supo = False
            self.expo = 0
            self.mant = 0
        else:
            if value.mantissa = 0:
                self.sign = value.exponent != 0
                self.supo = 0
                self.expo = 0
                self.mant = 0
                return

            if value.mantissa < 0:
                self.sign = True
                self.mant = -value.mantissa
            else:
                self.sign = False
                self.mant = value.mantissa

            if value.exponent < 0:
                self.supu = False
                self.expo = ~self.exponent
            else
                self.supu = True
                self.expo = self.exponent

            #FIXME
            pass

    def trunc(self, nbits):
        pass
