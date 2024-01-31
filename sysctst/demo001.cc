
#include <systemc.h>

SC_MODULE(test)
{
	sc_in<sc_logic> clk, ena, rst;
	sc_in<sc_lv<4>> in;
	sc_out<sc_lv<4>> out1;
	sc_out<sc_lv<4>> out2;

	sc_signal<sc_lv<4>> tmp;

	void method1()
	{
		if (rst.read() == SC_LOGIC_1)
			out1.write(0);
		else if (ena.read() == SC_LOGIC_1)
			out1.write(in.read());
	}

	void method2()
	{
		tmp.write(in.read());
	}

	void method3()
	{
		out2.write(tmp.read());
	}

	void dump()
	{
		cout << "@" << sc_time_stamp()
				<< " clk=" << clk << " ena=" << ena << " rst=" << rst << " in=" << in
				<< " out1=" << out1 << " out2=" << out2 << endl;
	}

	SC_CTOR(test)
	{
		SC_METHOD(method1);
		sensitive << clk.pos();

		SC_METHOD(method2);
		sensitive << in;

		SC_METHOD(method3);
		sensitive << tmp;
	}
};

int sc_main(int argc, char **argv)
{
	sc_signal<sc_logic> clk, ena, rst;
	sc_signal<sc_lv<4>> in;
	sc_signal<sc_lv<4>> out1;
	sc_signal<sc_lv<4>> out2;

	test uut("uut");
	uut.clk(clk);
	uut.rst(rst);
	uut.ena(ena);
	uut.in(in);
	uut.out1(out1);
	uut.out2(out2);

	sc_trace_file *wf = sc_create_vcd_trace_file("demo001");
	sc_trace(wf, clk, "clk");
	sc_trace(wf, ena, "ena");
	sc_trace(wf, rst, "rst");
	sc_trace(wf, in, "in");
	sc_trace(wf, out1, "out1");
	sc_trace(wf, out2, "out2");

	sc_start(1.5, SC_SEC);
	uut.dump();

#define CYCLE() do { clk.write(SC_LOGIC_0); sc_start(0.5, SC_SEC); clk.write(SC_LOGIC_1); sc_start(0.5, SC_SEC); uut.dump(); } while (0)
	rst.write(SC_LOGIC_1);
	ena.write(SC_LOGIC_1);
	CYCLE();

	rst.write(SC_LOGIC_0);
	ena.write(SC_LOGIC_1);
	CYCLE();

	in.write(0);
	CYCLE();

	in.write(3);
	CYCLE();

	in.write(7);
	CYCLE();

	ena.write(SC_LOGIC_0);
	in.write(0);
	CYCLE();

	CYCLE();
	sc_close_vcd_trace_file(wf);
	return 0;
}

