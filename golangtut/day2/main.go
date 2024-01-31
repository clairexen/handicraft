package main

import "fmt"
import "strings"
import "http"
import "net"
import "os"

type MyServer struct{}

func (this *MyServer) ServeHTTP(c *http.Conn, r *http.Request) {
	fileName := "." + r.URL.String();
	fmt.Println(r.Method, r.URL, "->", fileName);

	d, err := os.Stat(fileName);
	if err != nil {
		fmt.Println("Can't stat file:", err);
		goto error;
	}

	f, err := os.Open(fileName, os.O_RDONLY, 0);
	defer f.Close();
	if err != nil {
		fmt.Println("Can't open file:", err);
		goto error;
	}

	if d.IsDirectory() {
		c.SetHeader("Content-Type", "text/html; charset=utf-8");
		fmt.Fprintf(c, "<ul>\n");
		for {
			dirs, err := f.Readdir(100);
			if err != nil || len(dirs) == 0 {
				break
			}
			for _, e := range dirs {
				name := e.Name;
				if e.IsDirectory() {
					name += "/"
				}
				fmt.Fprintf(c, "<li><a href=\"%s\">%s</a></li>\n", name, name);
			}
		}
		fmt.Fprintf(c, "</ul>\n");
		return;
	}

	c.SetHeader("Content-Type", "text/plain; charset=utf-8");
	c.Write(strings.Bytes("--- BEGIN ROT13 ---\n"));

	for {
		var buf [1]byte;
		n, err := f.Read(&buf);
		if err != nil || n != 1 {
			break
		}
		switch {
		case 'a' <= buf[0] && buf[0] <= 'z':
			buf[0] = 'a' + ((buf[0]-'a')+13)%(1+'z'-'a')
		case 'A' <= buf[0] && buf[0] <= 'Z':
			buf[0] = 'A' + ((buf[0]-'A')+13)%(1+'Z'-'A')
		}
		c.Write(&buf);
	}

	c.Write(strings.Bytes("--- END ROT13 ---\n"));
	return;

error:
	c.SetHeader("Content-Type", "text/html; charset=utf-8");
	c.WriteHeader(404);
	c.Write(strings.Bytes("I/O Error!"));
	return;
}

func main() {
	fmt.Println("Starting server..");

	a, err := net.ResolveTCPAddr("127.0.0.1:8080");
	if err != nil {
		fmt.Println("Error in net.ResolveTCPAddr():", err);
		os.Exit(1);
	}

	l, err := net.ListenTCP("tcp", a);
	if err != nil {
		fmt.Println("Error in net.ListenTCP():", err);
		os.Exit(1);
	}

	h := new(MyServer);

	err = http.Serve(l, h);
	if err != nil {
		fmt.Println("Error in http.Serve():", err);
		os.Exit(1);
	}

	fmt.Println("Never reached?");
}
