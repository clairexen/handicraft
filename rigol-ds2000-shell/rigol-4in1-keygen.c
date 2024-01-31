/*
** rigol keygen / cybernet & the-eevblog-users
**
** to compile this you need MIRACL from [url]https://github.com/CertiVox/MIRACL[/url]
** download the master.zip into a new folder and run 'unzip -j -aa -L master.zip'
** then run 'bash linux' to build the miracle.a library
**
** BUILD WITH:
**
** more info: http://www.eevblog.com/forum/testgear/sniffing-the-rigol's-internal-i2c-bus/
*/

#define DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include "miracl.h"

// ECC SETTINGS
char DP832_private_key[]   = "5C393C30FACCF4";
char DS2000_private_key[]  = "8EEBD4D04C3771";
char DSA815_private_key[]  = "80444DFECE903E";
char DS1000Z_private_key[] = "6F1106DDA994DA";
char private_key[]         = "";
char prime1[]  = "AEBF94CEE3E707";
char prime2[]  = "AEBF94D5C6AA71";
char curve_a[] = "2982";
char curve_b[] = "3408";
char point1[]  = "7A3E808599A525";
char point2[]  = "28BE7FAFD2A052";


/*
** take serial and options make sha1 hash out of it
*/
static void hashing(char *opt_str, big hash) {
    char *p;
    char h[20];
    int ch;
    sha sh;
    shs_init(&sh);
    p = opt_str;

    while(*p) {
        shs_process(&sh, *p);
        p++;
    }

    shs_hash(&sh, h);
    bytes_to_big(20, h, hash);
}

/*
** sign the secret message (serial + opts) with the private key
*/
void ecssign(char *serial, char *options, char *privk, char *lic1, char *lic2) {
    int k_offset = 0; // optionally change ecssign starting offset (changes lic1; makes different licenses)
    mirsys(800, 16)->IOBASE = 16;

    sha sha1;
    shs_init(&sha1);

    char *ptr = serial;
    while(*ptr) shs_process(&sha1, *ptr++);
    ptr = options;
    while(*ptr) shs_process(&sha1, *ptr++);

    char h[20];
    shs_hash(&sha1, h);
    big hash = mirvar(0);
    bytes_to_big(20, h, hash);

    big a = mirvar(0);
    instr(a, curve_a);
    big b = mirvar(0);
    instr(b, curve_b);
    big p = mirvar(0);
    instr(p, prime1);
    big q = mirvar(0);
    instr(q, prime2);
    big Gx = mirvar(0);
    instr(Gx, point1);
    big Gy = mirvar(0);
    instr(Gy, point2);
    big d = mirvar(0);
    instr(d, privk);
    big k = mirvar(0);
    big r = mirvar(0);
    big s = mirvar(0);
    big k1 = mirvar(0);
    big zero = mirvar(0);

    big f1 = mirvar(17);
    big f2 = mirvar(53);
    big f3 = mirvar(905461);
    big f4 = mirvar(60291817);

    incr(k, k_offset, k);

    epoint *G = epoint_init();
    epoint *kG = epoint_init();
    ecurve_init(a, b, p, MR_PROJECTIVE);
    epoint_set(Gx, Gy, 0, G);

    for(;;) {
        incr(k, 1, k);

        if(divisible(k, f1) || divisible(k, f2) || divisible(k, f3) || divisible(k, f4))
            continue;

        ecurve_mult(k, G, kG);
        epoint_get(kG, r, r);
        divide(r, q, q);

        if(mr_compare(r, zero) == 0)
            continue;

        xgcd(k, q, k1, k1, k1);
        mad(d, r, hash, q, q, s);
        mad(s, k1, k1, q, q, s);

        if(!divisible(s, f1) && !divisible(s, f2) && !divisible(s, f3) && !divisible(s, f4))
            break;
    }

    cotstr(r, lic1);
    cotstr(s, lic2);
}

/*
** convert string to uppercase chars
*/
char * strtoupper(char *str) {
    char *p;
    for (p=str; *p; p++)
        *p = toupper(*p); 
    return str;
}

/*
** prepend a char to a string
*/
char * prepend(char *c, char *str) {
    int i;

    for (i = strlen(str); i >= 0; i--) {
        str[i + 1] = str[i];
    }

    str[0] = *c;
    return c;
}

/*
** convert hex-ascii-string to rigol license format
*/
void map_hex_to_rigol(char *io) {
    unsigned long long b = 0;
    int i = 0;
    char map[] = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
        'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
        'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '2', '3', '4', '5', '6', '7', '8', '9'
    };

    /* hex2dez */
    while (io[i] != '\0') {
        if (io[i] >= '0' && io[i] <= '9') {
            b = b * 16 + io[i] - '0';
        } else if (io[i] >= 'A' && io[i] <= 'F') {
            b = b * 16 + io[i] - 'A' + 10;
        } else if (io[i] >= 'a' && io[i] <= 'f') {
            b = b * 16 + io[i] - 'a' + 10;
        }
        i++;
    }

    for (i = 3; ; i--) {
        io[i] = map[b & 0x1F];
        if (i == 0) break;
        b >>= 5;
    }

    io[4] = '\0';
}

void show_help(char *cmd) {
    printf("Usage: %s <sn> <opts> <privkey>\n", cmd);
    printf("  <sn>       serial number of device (D............)\n");
    printf("  <opts>     device options, 4 characters, see below\n");
    printf("  <privkey>  private key (optional)\n");
    printf("\n");
    printf("DP832 device options:\n");
    printf("  first character:  M = official, 5 = trial\n");
    printf("  MWSS - Trigger\n");
    printf("  MWTB - Accuracy\n");
    printf("  MWTC - LAN and RS232\n");
    printf("  MWTE - Analyzer and Monitor\n");
    printf("\n");
    printf("DS1000z device options:\n");
    printf("  DSAB - Advanced Triggers\n");
    printf("  DSAC - Decoders\n");
    printf("  DSAE - 24M Memory\n");
    printf("  DSAJ - Recorder\n");
    printf("  DSBA - 500uV Vertical\n"); 
    printf("\n");
    printf("DS2000 device options:\n");
    printf("  first character:  D = official, V = trial\n");
    printf("  DSAB - Advanced Triggers\n");
    printf("  DSAC - Decoders\n");
    printf("  DSAE - 56M Memory\n");
    printf("  DSAJ - 100MHz\n");
    printf("  DSAS - 200MHz\n");
    printf("  DSAZ - all options\n");
    printf("\n");
    printf("DS4000 device options:\n");
    printf("  first character:  D = official, V = trial\n");
    printf("  DSHB - RS232 Decoder\n");
    printf("  DSHC - SPI Decoder\n");
    printf("  DSHE - I2C Decoder\n");
    printf("  DSHJ - CAN Decode\n");
    printf("  DSHS - FlexRay Decoder\n");
    printf("  DSA9 - all options\n");
    printf("\n");
    printf("DSA815 device options:\n");
    printf("  first character:  A = official, S = trial\n");
    printf("  AAAB - Tracking Generator\n");
    printf("  AAAC - Advnced Measurement Kit\n");
    printf("  AAAD - 10Hz RBW\n");
    printf("  AAAE - EMI/Quasi Peak\n");
    printf("  AAAF - VSWR\n");
    printf("\n");
    printf("MAKE SURE YOUR FIRMWARE IS UP TO DATE BEFORE APPLYING ANY KEYS\n");
}

char * make_licence(char *serial, char *options, char* priv_key)
{
    char *lic1_code, *lic2_code, *lic_all;
    char *chunk, *temp, *licence;
    int i, j;

    /* convert string to uppercase chars */
    strtoupper(serial);
    strtoupper(options);
    strtoupper(priv_key);

    /* sign the message */
    lic1_code = calloc(64, 1);
    lic2_code = calloc(64, 1);
    ecssign(serial, options, priv_key, lic1_code, lic2_code);

    /* fix missing zeroes */
    while (strlen(lic1_code) < 14) {
        prepend("0", lic1_code);
    }
    while (strlen(lic2_code) < 14) {
        prepend("0", lic2_code);
    }

    /* combine lic1 and lic2 */
    lic_all = calloc(128, 1);
    temp = calloc(128, 1);
    chunk = calloc(6, 1);
    strcpy(lic_all, lic1_code);
    strcat(lic_all, "0");
    strcat(lic_all, lic2_code);
    strcat(lic_all, "0");

    /* generate serial */
    i=0; 
    while (i < strlen(lic_all)) {
        memcpy(chunk, lic_all + i, 5);
        map_hex_to_rigol(chunk);
        strcat(temp, chunk);
        i = i + 5;
    }

 #ifdef DEBUG
    printf("lic1-code:        %s\n", lic1_code);
    printf("lic2-code:        %s\n", lic2_code);
    printf("target-code:      %s\n", lic_all);
#endif 

    /* add options and "-" */
    licence = calloc(128, 1);
    j = 0;
    for(i = 0; i <= strlen(temp); ) {
       switch(j) {
         case 1:  licence[j] = options[0];
		  break;
         case 7:  licence[j] = '-';
		  break;
         case 10: licence[j] = options[1];
		  break;
         case 15: licence[j] = '-';
		  break;
         case 19: licence[j] = options[2];
		  break;
         case 23: licence[j] = '-';
		  break;
         case 28: licence[j] = options[3];
		  break;
         default: licence[j] = temp[i];
                  i++;
       }
       j++;
    }
    licence[j] = '\0';

    /* cleen up */
    free(lic1_code);
    free(lic2_code);
    free(lic_all);
    free(chunk);
    free(temp);

    return licence;
}

/*
** the world ends here
*/
int main(int argc, char *argv[0]) {
    char *serial, *options, *priv_key, *licence;   
    int i=0, j=0;

    /* parse input */
    if (!((argc == 3 || argc == 4))) {
        show_help(argv[0]);
        exit(1);
    }
    serial = argv[1];
    options = argv[2];
    if (argc == 4) priv_key = argv[3];
    else if (!strncmp(serial, "DS1", 3)) priv_key = DS1000Z_private_key;
    else if (!strncmp(serial, "DS2", 3)) priv_key = DS2000_private_key;
    else if (!strncmp(serial, "DS4", 3)) priv_key = DS2000_private_key;
    else if (!strncmp(serial, "DSA", 3)) priv_key = DSA815_private_key;
    else if (!strncmp(serial, "DP8", 3)) priv_key = DP832_private_key;
    else {
        show_help(argv[0]);
        printf("\nERROR: UNKNOW DEVICE WITHOUT PRIVATKEY\n");
        exit(1);
    }

    if (strlen(priv_key) != 14) {
        show_help(argv[0]);
        printf("\nERROR: INVALID PRIVATE KEY LENGTH\n");
        exit(1);
    }
    if (strlen(serial) < 13) {
        show_help(argv[0]);
        printf("\nERROR: INVALID SERIAL LENGTH\n");
        exit(1);
    }
    if (strlen(options) != 4) {
        show_help(argv[0]);
        printf("\nERROR: INVALID OPTIONS LENGTH\n");
        exit(1);
    }

#ifdef DEBUG
    printf("private-key:      %s\n", priv_key);
    printf("serial:           %s\n", serial);
    printf("options:          %s\n", options);
#endif
    licence = make_licence(serial, options, priv_key);
#ifdef DEBUG
    printf("-------------------------------------------------\n");
    printf("your-license-key: %s\n", licence);
    printf("-------------------------------------------------\n");
#else
    printf("%s\n", licence);
#endif
    free(licence);
}

