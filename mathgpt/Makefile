mathgpt-tunedata.json: mathgpt.py training/*.mathgpt
	python3 mathgpt.py tunedata training/*.mathgpt > mathgpt-tunedata.json.new
	mv mathgpt-tunedata.json.new mathgpt-tunedata.json

fine-tune: mathgpt-tunedata.json
	openai api fine_tunes.create -t mathgpt-tunedata.json -m davinci
