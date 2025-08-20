
.PHONY: quickstart data features train evaluate app test

quickstart: data features train evaluate

data:
	python -m src.generate_data

features:
	python -m src.build_features

train:
	python -m src.train

evaluate:
	python -m src.evaluate

app:
	streamlit run app/Home.py

test:
	pytest -q