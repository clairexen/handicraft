NIXPY = nix-shell -p python39
PYC = cpython-39.pyc

compile:
	$(NIXPY) --run "python3 -m compileall *.py"
	mcp -dv '__pycache__/*.cpython-*.pyc' '#1.pyc'

001-ten-guesses:
	$(NIXPY) --run "python3 001-ten-guesses.py"

002-tic-tac-toe:
	$(NIXPY) --run "python3 002-tic-tac-toe.py"

checkout-flame:
	git clone git@github.com:FlameNotpartofthegame/LearningPython.git

update-flame: compile
	git -C LearningPython pull
	cp 001-ten-guesses.py LearningPython/numberguess_claire.py
	cp 002-tic-tac-toe.pyc LearningPython/tictactoe_claire.pyc
	git -C LearningPython commit -asm "Update from claire's repo"
	git -C LearningPython push

clean:
	rm -rf __pycache__ *.pyc
