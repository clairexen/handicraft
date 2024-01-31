
LDLIBS += -lstdc++ -lfltk
CXXFLAGS += -Wall -Wextra -Os -ggdb -MD
CXX = clang

DEMOS = demo001 demo002

all: $(DEMOS)

demo%: demo%.o
	$(CXX) $(LDFLAGS) -o $@ $< $(LDLIBS)

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(DEMOS)
	rm -f *.d *.o core

.PHONY: all clean

-include *.d

