NOTEBOOKS := $(sort $(wildcard notebooks/*.ipynb))

.PHONY: all data notebooks test clean

all: data notebooks test

data:
	python -m pizza_ab.simulate

notebooks: data
	@for nb in $(NOTEBOOKS); do \
		echo "executing $$nb"; \
		jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 $$nb || exit 1; \
	done

test:
	python -m pytest -q

clean:
	rm -rf data/*.csv .pytest_cache
