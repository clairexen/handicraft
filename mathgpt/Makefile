mathgpt-tunedata.json: mathgpt.py training/*.mathgpt
	python3 mathgpt.py tunedata mathgpt-tunedata.json.new training/*.mathgpt
	mv mathgpt-tunedata.json.new mathgpt-tunedata.json

fine-tune: mathgpt-tunedata.json
	openai api fine_tunes.create -t mathgpt-tunedata.json -m davinci
