all: test_add.log57 test_add.log58 test_add.log
	egrep 'VERIFICATION (SUCCESSFUL|FAILED)' test_add.log57 test_add.log58 test_add.log

cbmc57:
	wget -O cbmc-5-7-linux-64.tgz https://www.cprover.org/cbmc/download/cbmc-5-7-linux-64.tgz
	tar xzf cbmc-5-7-linux-64.tgz cbmc
	mv cbmc cbmc57

cbmc58: cbmc57
	wget -O cbmc-5-8-linux-64.tgz https://www.cprover.org/cbmc/download/cbmc-5-8-linux-64.tgz
	tar xzf cbmc-5-8-linux-64.tgz cbmc
	mv cbmc cbmc58

cbmc-git/src/cbmc/cbmc:
	git clone https://github.com/diffblue/cbmc-git
	$(MAKE) -C cbmc-git/src minisat2-download
	$(MAKE) -C cbmc-git/src

test_add.log57: cbmc57 test_add.cc test_add.h common.h encoding.h add.h
	time ./cbmc57 --trace --stop-on-fail --no-built-in-assertions --function test_add test_add.cc | \
			ts -s '%H:%M:%S [add]' | tee test_add.log57.new
	mv test_add.log57.new test_add.log57

test_add.log58: cbmc58 test_add.cc test_add.h common.h encoding.h add.h
	time ./cbmc58 --trace --stop-on-fail --no-built-in-assertions --function test_add test_add.cc | \
			ts -s '%H:%M:%S [add]' | tee test_add.log58.new
	mv test_add.log58.new test_add.log58

test_add.log: cbmc-git/src/cbmc/cbmc test_add.cc test_add.h common.h encoding.h add.h
	time cbmc-git/src/cbmc/cbmc --trace --stop-on-fail --no-built-in-assertions --function test_add test_add.cc | \
			ts -s '%H:%M:%S [add]' | tee test_add.log.new
	mv test_add.log.new test_add.log

clean:
	rm -f test_add.log57.new test_add.log57
	rm -f test_add.log58.new test_add.log58
	rm -f test_add.log.new test_add.log
