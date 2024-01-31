#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <sys/socket.h>

#define RPC_PORTMAP_PORT 111

#define RPC_PORTMAPPER_GETPORT 3
#define RPC_PORTMAPPER_PROGRAM 100000
#define RPC_PORTMAPPER_VERSION 2

#define RPC_VXI11_PROGRAM 0x0607AF
#define RPC_VXI11_VERSION 1

#define RPC_VXI11_CORE_PROGRAM 0x0607AF
#define RPC_VXI11_CORE_VERSION 1

#define RPC_VXI11_CORE_CREATE_LINK 10
#define RPC_VXI11_CORE_DEVICE_WRITE 11
#define RPC_VXI11_CORE_DEVICE_READ 12
#define RPC_VXI11_CORE_DESTROY_LINK 23

#define RPC_BUF_WRITE(buffer, value) do { *buffer++ = htonl((value)); } while(0)

#define RPC_BUF_WRITE_OPAQUE(buffer, data, datalen) do { RPC_BUF_WRITE(buffer, datalen); memcpy(buffer, data, datalen); buffer += (datalen+3)/4; } while (0)

#define RPC_HDR_WRITE(buffer, XID, program, version, call) do { RPC_BUF_WRITE(buffer, XID); \
                                                                RPC_BUF_WRITE(buffer, 0); \
                                                                RPC_BUF_WRITE(buffer, 2); \
                                                                RPC_BUF_WRITE(buffer, program);\
                                                                RPC_BUF_WRITE(buffer, version);\
                                                                RPC_BUF_WRITE(buffer, call);\
                                                                RPC_BUF_WRITE(buffer, 0); \
                                                                RPC_BUF_WRITE(buffer, 0); \
                                                                RPC_BUF_WRITE(buffer, 0); \
                                                                RPC_BUF_WRITE(buffer, 0); } while(0)

#define RPC_FRAG_WRITE(msg, pointer) (*((uint32_t*)msg) = htonl(0x80000000 | (((uint32_t)((uint8_t*)pointer-(uint8_t*)msg))-4)), (((uint32_t)((uint8_t*)pointer-(uint8_t*)msg))))
#define RPC_HDR_IS_REPLY(msg, XID) ((ntohl(msg[0])==XID) && (ntohl(msg[1])==1) && (ntohl(msg[2])==0) && (ntohl(msg[3])==0) && (ntohl(msg[5])==0))

static int vxi11_getport(uint32_t ip)
{
  struct sockaddr_in sin;
  int fd;

  // Connect
  fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  
  memset((char *) &sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons(RPC_PORTMAP_PORT);
  sin.sin_addr.s_addr = ip;
  if (connect(fd, (struct sockaddr*)&sin, sizeof(sin)) == -1)
  {
    close(fd);
    return -1;
  }
  
  // Send request
  uint32_t buf[64];
  uint32_t *hdr = buf;
  
  uint32_t XID = 123; 
  
  hdr++; // Alloc fragment header
  RPC_HDR_WRITE(hdr, XID, RPC_PORTMAPPER_PROGRAM, RPC_PORTMAPPER_VERSION, RPC_PORTMAPPER_GETPORT);
  // Data
  RPC_BUF_WRITE(hdr, RPC_VXI11_PROGRAM);
  RPC_BUF_WRITE(hdr, RPC_VXI11_VERSION);
  RPC_BUF_WRITE(hdr, 6); // Protocol = TCP
  RPC_BUF_WRITE(hdr, 0); // Port
  
  uint32_t msgSize = RPC_FRAG_WRITE(buf, hdr);
  
  if (send(fd, buf, msgSize, 0) == -1) { close(fd); return -2; }
 
  // Read response
  int read;
  if ((read = recv(fd, buf, 4, 0)) == -1) { close(fd); return -3; }
  
  // Parse fragment header
  uint32_t frag = ntohl(buf[0]);
  if (!(frag & 0x80000000)) { close(fd); return -4; }
  
  int toRead = frag & 0x7FFFFFFF;
  
  if (toRead > 64*4) { close(fd); return -4; }
  
  if (recv(fd, buf, toRead, 0) != toRead) { close(fd); return -5; }
  if (!RPC_HDR_IS_REPLY(buf, XID)) { close(fd); return -6; }
  
  uint32_t port = ntohl(buf[6]);
  
  close(fd);
  return port;
}

typedef struct 
{
  int fd;
  uint32_t link;
  uint32_t xid;
} vxi11_conn;

int vxi11_open(vxi11_conn* conn, uint32_t ip, const char* instrument)
{
  int len = strlen(instrument);
  if ((len <= 0) || (len > 32)) return -10; // Never seen anything more than 10-16 characters

  // Get port
  int port = vxi11_getport(ip);
  if(port<0) return port;
  
  // Create socket
  conn->fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  conn->xid = 0;
  conn->link = 0;
  
  // Connect
  struct sockaddr_in sin;
  
  memset((char *) &sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = ip;
  if (connect(conn->fd, (struct sockaddr*)&sin, sizeof(sin)) == -1)
  {
    close(conn->fd);
    return -1;
  }
  
  // Call create_link
  uint32_t buf[64];
  uint32_t *hdr = &buf[1];
  uint32_t xid = conn->xid++;
  
  RPC_HDR_WRITE(hdr, xid, RPC_VXI11_CORE_PROGRAM, RPC_VXI11_CORE_VERSION, RPC_VXI11_CORE_CREATE_LINK);
  RPC_BUF_WRITE(hdr, 123); // Client ID
  RPC_BUF_WRITE(hdr, 0); // Lock device
  RPC_BUF_WRITE(hdr, 0); // Lock timeout
  RPC_BUF_WRITE_OPAQUE(hdr, instrument, len);
  
  uint32_t msgSize = RPC_FRAG_WRITE(buf, hdr);
  
  if (send(conn->fd, buf, msgSize, 0) == -1) { close(conn->fd); return -2; }
  
  // Read response
  int read;
  if ((read = recv(conn->fd, buf, 4, 0)) == -1) { close(conn->fd); return -3; }
  
  // Parse fragment header
  uint32_t frag = ntohl(buf[0]);
  if (!(frag & 0x80000000)) { close(conn->fd); return -4; }
  
  int toRead = frag & 0x7FFFFFFF;
  
  if (toRead > 64*4) { close(conn->fd); return -5; }
  
  if (recv(conn->fd, buf, toRead, 0) != toRead) { close(conn->fd); return -6; }
  if (!RPC_HDR_IS_REPLY(buf, xid)) { close(conn->fd); return -7; }
  
  uint32_t error = ntohl(buf[6]);
  conn->link = ntohl(buf[7]);
  
  if(error!=0)
  {
    close(conn->fd);
    return error;
  }
  
  return 0;
}

int vxi11_close(vxi11_conn* conn)
{
  // Call destroy_link
  uint32_t buf[64];
  uint32_t *hdr = &buf[1];
  uint32_t xid = conn->xid++;
  
  RPC_HDR_WRITE(hdr, xid, RPC_VXI11_CORE_PROGRAM, RPC_VXI11_CORE_VERSION, RPC_VXI11_CORE_DESTROY_LINK);
  RPC_BUF_WRITE(hdr, conn->link); // lid
  
  uint32_t msgSize = RPC_FRAG_WRITE(buf, hdr);
  
  if (send(conn->fd, buf, msgSize, 0) == -1) return -2;
  
  // Read response, just because we are nice
  int read;
  if ((read = recv(conn->fd, buf, 4, 0)) == -1) return -3;
  
  // Parse fragment header
  uint32_t frag = ntohl(buf[0]);
  if (!(frag & 0x80000000)) return -4;
  
  int toRead = frag & 0x7FFFFFFF;
  
  if (toRead > 64*4) return -5;
  
  if (recv(conn->fd, buf, toRead, 0) != toRead) return -6;
  if (!RPC_HDR_IS_REPLY(buf, xid)) return -7;
  
  uint32_t error = ntohl(buf[6]);
  
  close(conn->fd);
  return error;
}

int vxi11_write(vxi11_conn* conn, const void* data, int data_len, int eoi)
{
  // Call destroy_link
  uint32_t buf[16+data_len/4];
  uint32_t *hdr = &buf[1];
  uint32_t xid = conn->xid++;
  
  RPC_HDR_WRITE(hdr, xid, RPC_VXI11_CORE_PROGRAM, RPC_VXI11_CORE_VERSION, RPC_VXI11_CORE_DEVICE_WRITE);
  RPC_BUF_WRITE(hdr, conn->link); // lid
  RPC_BUF_WRITE(hdr, 10000); // io_timeout
  RPC_BUF_WRITE(hdr, 0); // lock_timeout
  RPC_BUF_WRITE(hdr, (eoi ? 0x8 : 0)); // flags
  RPC_BUF_WRITE_OPAQUE(hdr, data, data_len); 
  
  uint32_t msgSize = RPC_FRAG_WRITE(buf, hdr);
  
  if (send(conn->fd, buf, msgSize, 0) == -1) return -2;
  
  // Read response
  int read;
  if ((read = recv(conn->fd, buf, 4, 0)) == -1) return -3;
  
  // Parse fragment header
  uint32_t frag = ntohl(buf[0]);
  if (!(frag & 0x80000000)) return -4;
  
  int toRead = frag & 0x7FFFFFFF;
  
  if (toRead > 64*4) return -5;
  
  if (recv(conn->fd, buf, toRead, 0) != toRead) return -6;
  if (!RPC_HDR_IS_REPLY(buf, xid)) return -7;
  
  uint32_t error = ntohl(buf[6]);
  uint32_t bytes_written = ntohl(buf[7]);
  
  if (error != 0)
    return -error;
  else
    return bytes_written;
}

int vxi11_read(vxi11_conn* conn, void* buffer, int buffer_len, int* eoi)
{
  // Call destroy_link
  uint32_t buf[16+buffer_len/4];
  uint32_t *hdr = &buf[1];
  uint32_t xid = conn->xid++;
  
  RPC_HDR_WRITE(hdr, xid, RPC_VXI11_CORE_PROGRAM, RPC_VXI11_CORE_VERSION, RPC_VXI11_CORE_DEVICE_READ);
  RPC_BUF_WRITE(hdr, conn->link); // lid
  RPC_BUF_WRITE(hdr, buffer_len); // io_timeout
  RPC_BUF_WRITE(hdr, 10000); // io_timeout
  RPC_BUF_WRITE(hdr, 0); // lock_timeout
  RPC_BUF_WRITE(hdr, 0); // flags
  RPC_BUF_WRITE(hdr, 0); // term_char
  
  uint32_t msgSize = RPC_FRAG_WRITE(buf, hdr);
  
  if (send(conn->fd, buf, msgSize, 0) == -1) return -2;
  
  // Read response
  int read;
  if ((read = recv(conn->fd, buf, 4, 0)) == -1) return -3;
  
  // Parse fragment header
  uint32_t frag = ntohl(buf[0]);
  if (!(frag & 0x80000000)) return -4;
  
  int toRead = frag & 0x7FFFFFFF;
  
  if (toRead < 9*4) return -5;
  
  if (recv(conn->fd, buf, 9*4, 0) != 9*4) return -6;
  if (!RPC_HDR_IS_REPLY(buf, xid)) return -7;
  
  uint32_t error = ntohl(buf[6]);
  uint32_t reason = ntohl(buf[7]);
  uint32_t data_len = ntohl(buf[8]);
  
  if (data_len <= 0)
    return -error;
  else
  {
    if (recv(conn->fd, buffer, data_len, 0) != data_len) return -8;
    
    // Read the last bits to align the stream to 32bits
    int left = 4-(data_len&3);
    if (left != 0)
      if (recv(conn->fd, buf, left, 0) != left) return -9;
    
    *eoi = ((reason & 0x4)!=0);
    return data_len;
  }
}

int main(int argc, char **argv)
{
  uint32_t ip = 1234567;
  inet_pton(AF_INET, argv[1], &ip);

  vxi11_conn conn;
  
  printf("Create link: %d\n", vxi11_open(&conn, ip, "inst0"));
  
  printf("Wrote: %d\n", vxi11_write(&conn, "*IDN?\n", 6, 1));
  
  char buf[1025];
  int eoi;
  int read = vxi11_read(&conn, buf, 1024, &eoi);
  
  printf("Read: %d (EOI=%d)\n", read, eoi);
  if(read > 0)
  {
    buf[read] = '\0';
    printf("Data: '%s'\n", buf);
  }
  
  printf("Destroy link: %d\n", vxi11_close(&conn));
  return 0;
}
