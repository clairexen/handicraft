
run: tr-fsm-syn
	./tr-fsm-syn

tr-fsm-syn: synth.pas lexer.pas parser.pas
	fpc -o$@ synth.pas

lexer.pas: lexer.l
	plex lexer.l

parser.pas: parser.y
	# pyacc -v -d parser.y
	pyacc parser.y

clean:
	rm -rf tr-fsm-syn core *.o
	rm -f lexer.pas parser.pas parser.lst

