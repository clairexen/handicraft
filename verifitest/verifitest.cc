#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <set>

#include "veri_file.h"
#include "vhdl_file.h"
#include "VeriWrite.h"
#include "DataBase.h"
#include "Message.h"

#ifdef VERIFIC_NAMESPACE
using namespace Verific;
#endif

void msg_func(msg_type_t msg_type, const char *message_id, linefile_type linefile, const char *msg, va_list args)
{
	printf("VERIFIC-%s [%s] ",
			msg_type == VERIFIC_NONE ? "NONE" :
			msg_type == VERIFIC_ERROR ? "ERROR" :
			msg_type == VERIFIC_WARNING ? "WARNING" :
			msg_type == VERIFIC_IGNORE ? "IGNORE" :
			msg_type == VERIFIC_INFO ? "INFO" :
			msg_type == VERIFIC_COMMENT ? "COMMENT" :
			msg_type == VERIFIC_PROGRAM_ERROR ? "PROGRAM_ERROR" : "UNKNOWN", message_id);
	if (linefile)
		printf("%s:%d: ", LineFile::GetFileName(linefile), LineFile::GetLineNo(linefile));
	vprintf(msg, args);
	printf("\n");
}

const char *type_by_id(int id)
{
#define X(_n) if (id == _n) return #_n;
	X(PRIM_NONE)
	X(PRIM_PWR)
	X(PRIM_GND)
	X(PRIM_X)
	X(PRIM_Z)
	X(PRIM_INV)
	X(PRIM_BUF)
	X(PRIM_AND)
	X(PRIM_NAND)
	X(PRIM_OR)
	X(PRIM_NOR)
	X(PRIM_XOR)
	X(PRIM_XNOR)
	X(PRIM_MUX)
	X(PRIM_PULLUP)
	X(PRIM_PULLDOWN)
	X(PRIM_TRI)
	X(PRIM_BUFIF1)
	X(PRIM_DLATCH)
	X(PRIM_DLATCHRS)
	X(PRIM_DFF)
	X(PRIM_DFFRS)
	X(PRIM_NMOS)
	X(PRIM_PMOS)
	X(PRIM_CMOS)
	X(PRIM_TRAN)
	X(PRIM_FADD)
	X(PRIM_RCMOS)
	X(PRIM_RNMOS)
	X(PRIM_RPMOS)
	X(PRIM_RTRAN)
	X(PRIM_HDL_ASSERTION)
	X(OPER_ADDER)
	X(OPER_MULTIPLIER)
	X(OPER_DIVIDER)
	X(OPER_MODULO)
	X(OPER_REMAINDER)
	X(OPER_SHIFT_LEFT)
	X(OPER_SHIFT_RIGHT)
	X(OPER_ROTATE_LEFT)
	X(OPER_ROTATE_RIGHT)
	X(OPER_REDUCE_AND)
	X(OPER_REDUCE_OR)
	X(OPER_REDUCE_XOR)
	X(OPER_REDUCE_NAND)
	X(OPER_REDUCE_NOR)
	X(OPER_REDUCE_XNOR)
	X(OPER_LESSTHAN)
	X(OPER_NTO1MUX)
	X(OPER_SELECTOR)
	X(OPER_DECODER)
	X(OPER_ENABLED_DECODER)
	X(OPER_PRIO_SELECTOR)
	X(OPER_DUAL_PORT_RAM)
	X(OPER_READ_PORT)
	X(OPER_WRITE_PORT)
	X(OPER_CLOCKED_WRITE_PORT)
	X(OPER_LUT)
	X(OPER_POW)
	X(OPER_PRIO_ENCODER)
	X(OPER_ABS)
	X(OPER_PSLPREV)
	X(OPER_PSLNEXTFUNC)
	X(PRIM_PSL_ASSERT)
	X(PRIM_PSL_ASSUME)
	X(PRIM_PSL_ASSUME_GUARANTEE)
	X(PRIM_PSL_RESTRICT)
	X(PRIM_PSL_RESTRICT_GUARANTEE)
	X(PRIM_PSL_COVER)
	X(PRIM_ENDPOINT)
	X(PRIM_ROSE)
	X(PRIM_FELL)
	X(PRIM_AT)
	X(PRIM_ATSTRONG)
	X(PRIM_ABORT)
	X(PRIM_PSL_NOT)
	X(PRIM_PSL_AND)
	X(PRIM_PSL_OR)
	X(PRIM_IMPL)
	X(PRIM_EQUIV)
	X(PRIM_PSL_X)
	X(PRIM_PSL_XSTRONG)
	X(PRIM_PSL_G)
	X(PRIM_PSL_F)
	X(PRIM_PSL_U)
	X(PRIM_PSL_W)
	X(PRIM_NEXT)
	X(PRIM_NEXTSTRONG)
	X(PRIM_ALWAYS)
	X(PRIM_NEVER)
	X(PRIM_EVENTUALLY)
	X(PRIM_UNTIL)
	X(PRIM_UNTIL_)
	X(PRIM_UNTILSTRONG)
	X(PRIM_UNTILSTRONG_)
	X(PRIM_BEFORE)
	X(PRIM_BEFORE_)
	X(PRIM_BEFORESTRONG)
	X(PRIM_BEFORESTRONG_)
	X(PRIM_NEXT_A)
	X(PRIM_NEXT_ASTRONG)
	X(PRIM_NEXT_E)
	X(PRIM_NEXT_ESTRONG)
	X(PRIM_NEXT_EVENT)
	X(PRIM_NEXT_EVENTSTRONG)
	X(PRIM_NEXT_EVENT_A)
	X(PRIM_NEXT_EVENT_ASTRONG)
	X(PRIM_NEXT_EVENT_E)
	X(PRIM_NEXT_EVENT_ESTRONG)
	X(PRIM_SEQ_IMPL)
	X(PRIM_OSUFFIX_IMPL)
	X(PRIM_SUFFIX_IMPL)
	X(PRIM_OSUFFIX_IMPLSTRONG)
	X(PRIM_SUFFIX_IMPLSTRONG)
	X(PRIM_WITHIN)
	X(PRIM_WITHIN_)
	X(PRIM_WITHINSTRONG)
	X(PRIM_WITHINSTRONG_)
	X(PRIM_WHILENOT)
	X(PRIM_WHILENOT_)
	X(PRIM_WHILENOTSTRONG)
	X(PRIM_WHILENOTSTRONG_)
	X(PRIM_CONCAT)
	X(PRIM_FUSION)
	X(PRIM_SEQ_AND_LEN)
	X(PRIM_SEQ_AND)
	X(PRIM_SEQ_OR)
	X(PRIM_CONS_REP)
	X(PRIM_NONCONS_REP)
	X(PRIM_GOTO_REP)
	X(OPER_WIDE_AND)
	X(OPER_WIDE_OR)
	X(OPER_WIDE_XOR)
	X(OPER_WIDE_NAND)
	X(OPER_WIDE_NOR)
	X(OPER_WIDE_XNOR)
	X(OPER_WIDE_BUF)
	X(OPER_WIDE_INV)
	X(OPER_WIDE_TRI)
	X(OPER_MINUS)
	X(OPER_UMINUS)
	X(OPER_EQUAL)
	X(OPER_NEQUAL)
	X(OPER_WIDE_MUX)
	X(OPER_WIDE_NTO1MUX)
	X(OPER_WIDE_SELECTOR)
	X(OPER_WIDE_DFF)
	X(OPER_WIDE_DFFRS)
	X(OPER_WIDE_DLATCHRS)
	X(OPER_WIDE_DLATCH)
	X(OPER_WIDE_PRIO_SELECTOR)
	X(PRIM_SVA_IMMEDIATE_ASSERT)
	X(PRIM_SVA_ASSERT)
	X(PRIM_SVA_COVER)
	X(PRIM_SVA_ASSUME)
	X(PRIM_SVA_EXPECT)
	X(PRIM_SVA_POSEDGE)
	X(PRIM_SVA_NOT)
	X(PRIM_SVA_FIRST_MATCH)
	X(PRIM_SVA_ENDED)
	X(PRIM_SVA_MATCHED)
	X(PRIM_SVA_CONSECUTIVE_REPEAT)
	X(PRIM_SVA_NON_CONSECUTIVE_REPEAT)
	X(PRIM_SVA_GOTO_REPEAT)
	X(PRIM_SVA_MATCH_ITEM_TRIGGER)
	X(PRIM_SVA_AND)
	X(PRIM_SVA_OR)
	X(PRIM_SVA_SEQ_AND)
	X(PRIM_SVA_SEQ_OR)
	X(PRIM_SVA_EVENT_OR)
	X(PRIM_SVA_OVERLAPPED_IMPLICATION)
	X(PRIM_SVA_NON_OVERLAPPED_IMPLICATION)
	X(PRIM_SVA_OVERLAPPED_FOLLOWED_BY)
	X(PRIM_SVA_NON_OVERLAPPED_FOLLOWED_BY)
	X(PRIM_SVA_INTERSECT)
	X(PRIM_SVA_THROUGHOUT)
	X(PRIM_SVA_WITHIN)
	X(PRIM_SVA_AT)
	X(PRIM_SVA_DISABLE_IFF)
	X(PRIM_SVA_SAMPLED)
	X(PRIM_SVA_ROSE)
	X(PRIM_SVA_FELL)
	X(PRIM_SVA_STABLE)
	X(PRIM_SVA_PAST)
	X(PRIM_SVA_MATCH_ITEM_ASSIGN)
	X(PRIM_SVA_SEQ_CONCAT)
	X(PRIM_SVA_IF)
	X(PRIM_SVA_RESTRICT)
	X(PRIM_SVA_TRIGGERED)
	X(PRIM_SVA_STRONG)
	X(PRIM_SVA_WEAK)
	X(PRIM_SVA_NEXTTIME)
	X(PRIM_SVA_S_NEXTTIME)
	X(PRIM_SVA_ALWAYS)
	X(PRIM_SVA_S_ALWAYS)
	X(PRIM_SVA_S_EVENTUALLY)
	X(PRIM_SVA_EVENTUALLY)
	X(PRIM_SVA_UNTIL)
	X(PRIM_SVA_S_UNTIL)
	X(PRIM_SVA_UNTIL_WITH)
	X(PRIM_SVA_S_UNTIL_WITH)
	X(PRIM_SVA_IMPLIES)
	X(PRIM_SVA_IFF)
	X(PRIM_SVA_ACCEPT_ON)
	X(PRIM_SVA_REJECT_ON)
	X(PRIM_SVA_SYNC_ACCEPT_ON)
	X(PRIM_SVA_SYNC_REJECT_ON)
	X(PRIM_SVA_GLOBAL_CLOCKING_DEF)
	X(PRIM_SVA_GLOBAL_CLOCKING_REF)
	X(PRIM_SVA_IMMEDIATE_ASSUME)
	X(PRIM_SVA_IMMEDIATE_COVER)
	X(OPER_SVA_SAMPLED)
	X(OPER_SVA_STABLE)
	X(OPER_WIDE_CASE_SELECT_BOX)
	X(PRIM_END)
	return "unknown";
}

void dump_common(FILE *f, DesignObj *obj)
{
	MapIter mi;
	Att *attr;

	if (obj->Linefile())
		fprintf(f, "    LINEFILE %s %d\n", LineFile::GetFileName(obj->Linefile()), LineFile::GetLineNo(obj->Linefile()));

	FOREACH_ATTRIBUTE(obj, mi, attr)
		fprintf(f, "    ATTRIBUTE %s = %s\n", attr->Key(), attr->Value());
}

void dump_netlist(FILE *f, Netlist *nl)
{
	MapIter mi, mi2;

	fprintf(f, "NETLIST: %s\n", nl->Owner()->Name());

	if (nl->IsPrimitive())
		fprintf(f, "  IS_PRIMITIVE\n");

	if (nl->IsOperator())
		fprintf(f, "  IS_OPERATOR\n");

	if (nl->IsConstant())
		fprintf(f, "  IS_CONSTANT\n");

	if (nl->IsAssertion())
		fprintf(f, "  IS_ASSERTATION\n");

	if (nl->IsCombinational())
		fprintf(f, "  IS_COMBINATIONAL\n");

	if (nl->IsBlackBox())
		fprintf(f, "  IS_BLACKBOX\n");

	Port *port;
	FOREACH_PORT_OF_NETLIST(nl, mi, port) {
		fprintf(f, "  PORT: %s\n", port->Name());
		dump_common(f, port);
		if (port->Bus())
			fprintf(f, "    BUS: %s %d\n", port->Bus()->Name(), port->Bus()->IndexOf(port));
		if (port->GetNet())
			fprintf(f, "    NET: %s\n", port->GetNet()->Name());
		if (port->GetDir() == DIR_INOUT)
			fprintf(f, "    INOUT\n");
		if (port->GetDir() == DIR_IN)
			fprintf(f, "    INPUT\n");
		if (port->GetDir() == DIR_OUT)
			fprintf(f, "    OUTPUT\n");
	}

	PortBus *portbus;
	FOREACH_PORTBUS_OF_NETLIST(nl, mi, portbus) {
		fprintf(f, "  PORTBUS: %s [%d:%d]\n", portbus->Name(), portbus->LeftIndex(), portbus->RightIndex());
		dump_common(f, portbus);
		if (port->GetDir() == DIR_INOUT)
			fprintf(f, "    INOUT\n");
		if (port->GetDir() == DIR_IN)
			fprintf(f, "    INPUT\n");
		if (port->GetDir() == DIR_OUT)
			fprintf(f, "    OUTPUT\n");
		for (int i = portbus->LeftIndex();; i += portbus->IsUp() ? +1 : -1) {
			fprintf(f, "    %3d: %s\n", i, portbus->ElementAtIndex(i) ? portbus->ElementAtIndex(i)->Name() : "");
			if (i == portbus->RightIndex())
				break;
		}
	}

	Net *net;
	FOREACH_NET_OF_NETLIST(nl, mi, net) {
		fprintf(f, "  NET: %s\n", net->Name());
		dump_common(f, net);
		if (net->Bus())
			fprintf(f, "    BUS: %s %d\n", net->Bus()->Name(), net->Bus()->IndexOf(net));
		if (net->GetInitialValue())
			fprintf(f, "    INIT: %c\n", net->GetInitialValue());
	}

	NetBus *netbus;
	FOREACH_NETBUS_OF_NETLIST(nl, mi, netbus) {
		fprintf(f, "  NETBUS: %s [%d:%d]\n", netbus->Name(), netbus->LeftIndex(), netbus->RightIndex());
		dump_common(f, netbus);
		for (int i = netbus->LeftIndex();; i += netbus->IsUp() ? +1 : -1) {
			fprintf(f, "    %3d: %s\n", i, netbus->ElementAtIndex(i) ? netbus->ElementAtIndex(i)->Name() : "");
			if (i == netbus->RightIndex())
				break;
		}
	}

	Instance *inst;
	FOREACH_INSTANCE_OF_NETLIST(nl, mi, inst) {
		PortRef *pr ;
		fprintf(f, "  INSTANCE: %s %s\n", inst->View()->Owner()->Name(), inst->Name());
		fprintf(f, "    TYPE: %s (%d)\n", type_by_id(inst->Type()), int(inst->Type()));
		if (inst->IsUserDeclared())
			fprintf(f, "    IS_USER_DECLARED\n");
		dump_common(f, inst);
		FOREACH_PORTREF_OF_INST(inst, mi2, pr)
			fprintf(f, "    PORTREF: %s %s\n", pr->GetPort()->Name(), pr->GetNet()->Name());
	}
}

int main(int argc, char **argv)
{
	int opt;
	const char *topmod = NULL;
	const char *outfile = NULL;
	const char *infofile = NULL;

	while ((opt = getopt(argc, argv, "t:o:i:")) != -1)
	{
		switch (opt)
		{
		case 't':
			topmod = optarg;
			break;
		case 'o':
			outfile = optarg;
			break;
		case 'i':
			infofile = optarg;
			break;
		default:
			goto help;
		}
	}

	if (optind == argc) {
help:
		fprintf(stderr, "Usage: %s [-t <top_module>] [-o <out_file>] [-i <info_file>] <infiles>\n", argv[0]);
		return 1;
	}

	Message::SetConsoleOutput(0);
	Message::RegisterCallBackMsg(msg_func);

	vhdl_file vhdl_reader;
	veri_file veri_reader;
	VeriWrite veri_writer;

	bool got_vhdl = false;
	bool got_veri = false;

	for (; optind < argc; optind++)
		if (strlen(argv[optind]) > 4 && !strcmp(argv[optind]+strlen(argv[optind])-4, ".vhd")) {
			if (!got_vhdl)
				vhdl_reader.SetDefaultLibraryPath(VERIFIC_DIR "/vhdl_packages/vdbs");
			if (!vhdl_reader.Analyze(argv[optind], "work", vhdl_file::VHDL_PSL)) {
				fprintf(stderr, "vhdl_reader.Analyze() failed for `%s'.\n", argv[optind]);
				return 1;
			}
			got_vhdl = true;
		} else {
			if (!veri_reader.Analyze(argv[optind], veri_file::SYSTEM_VERILOG)) {
				fprintf(stderr, "veri_reader.Analyze() failed for `%s'.\n", argv[optind]);
				return 1;
			}
			got_veri = true;
		}

	if (topmod == NULL)
	{
		if (got_vhdl && got_veri) {
			fprintf(stderr, "For mixed-langugage designs the -t option (specify top module) is mandatory.\n");
			return 1;
		}

		if (got_vhdl)
			if (!vhdl_reader.ElaborateAll()) {
				fprintf(stderr, "vhdl_reader.ElaborateAll() failed.\n");
				return 1;
			}

		if (got_veri)
			if (!veri_reader.ElaborateAll()) {
				fprintf(stderr, "veri_reader.ElaborateAll() failed.\n");
				return 1;
			}
	}
	else
	{
		if (veri_reader.GetModule(topmod))
		{
			if (!veri_reader.Elaborate(topmod)) {
				fprintf(stderr, "veri_reader.Elaborate(\"%s\") failed.\n", topmod);
				return 1;
			}
		}
		else
		{
			if (!vhdl_reader.Elaborate(topmod)) {
				fprintf(stderr, "vhdl_reader.Elaborate(\"%s\") failed.\n", topmod);
				return 1;
			}
		}
	}

	Netlist *top = Netlist::PresentDesign();
	if (!top) {
		fprintf(stderr, "Netlist::PresentDesign() failed.\n");
		return 1;
	}

	printf("INFO: Top-level module: %s\n", top->Owner()->Name());

	if (outfile != NULL) {
		veri_writer.WriteFile(outfile, top);
		if (infofile == NULL)
			return 0;
	}

	std::set<Netlist*> nl_todo, nl_done;
	nl_todo.insert(top);

	FILE *f = stdout;

	if (infofile)
		f = fopen(infofile, "w");

	while (!nl_todo.empty())
	{
		Netlist *nl = *nl_todo.begin();
		dump_netlist(f, nl);

		nl_todo.erase(nl);
		nl_done.insert(nl);

		MapIter mi;
		Instance *inst;
		FOREACH_INSTANCE_OF_NETLIST(nl, mi, inst) {
			if (!nl_done.count(inst->View()))
				nl_todo.insert(inst->View());
		}
	}

	printf("INFO: Memory usage before reset: %lu\n", Libset::Global()->MemUsage());
	Libset::Reset();

	return 0;
}

