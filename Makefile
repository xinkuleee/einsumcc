.PHONY: test demo check benchmark

PYTHON ?= python3 -B
export PYTHONPATH := src

test:
	$(PYTHON) -m unittest discover -s tests -v

demo:
	$(PYTHON) -m einsumcc explain 'aijd,bckd->abcijk' --lhs-shape 2,3,4,5 --rhs-shape 6,7,8,5

check: test
	$(PYTHON) -m einsumcc verify 'bij,bjk->bik' --lhs-shape 2,3,4 --rhs-shape 2,4,5 --seed 7

benchmark:
	@$(PYTHON) -m einsumcc benchmark benchmarks/cpu-smoke.json --repeats 3
