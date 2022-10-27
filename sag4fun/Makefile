run: sag4fun32c
	./sag4fun32c

sag4fun32c: testbench.v sag4fun32c.v
	iverilog -o $@ $^

clean:
	rm -f sag4fun32c
