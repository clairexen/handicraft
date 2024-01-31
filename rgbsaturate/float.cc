#include <math.h>

typedef struct {
    float r;       // percent
    float g;       // percent
    float b;       // percent
} rgb;

typedef struct {
    float h;       // angle in degrees
    float s;       // percent
    float v;       // percent
} hsv;

static void rgb2hsv(float &v1, float &v2, float &v3)
{
    rgb        in = { v1, v2, v3 };
    hsv        out;
    float      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if( max > 0.0 ) {
        out.s = (delta / max);                  // s
    } else {
        // r = g = b = 0                        // s = 0, v is undefined
        out.s = 0.0;
        out.h = 0; // NAN;                      // its now undefined
        goto ret;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

ret:
    v1 = out.h;
    v2 = out.s;
    v3 = out.v;
}

static void hsv2rgb(float &v1, float &v2, float &v3)
{
    hsv         in = { v1, v2, v3 };
    float       hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        if(isnan(in.h)) {   // in.h == NAN
            out.r = in.v;
            out.g = in.v;
            out.b = in.v;
            goto ret;
        }
        // error - should never happen
        out.r = 0.0;
        out.g = 0.0;
        out.b = 0.0;
        goto ret;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
ret:
    v1 = out.r;
    v2 = out.g;
    v3 = out.b;
}

void saturateRgb_float(float &r, float &g, float &b)
{
  rgb2hsv(r, g, b);
  g = 1;
  b = 1;
  hsv2rgb(r, g, b);
}


