.PHONY: test test-mlir test-cpu-codegen build-mlir demo check benchmark benchmark-native

PYTHON ?= python3 -B
export PYTHONPATH := src

test:
	$(PYTHON) -m unittest discover -s tests -v

build-mlir:
	./scripts/build-mlir.sh

test-mlir: build-mlir
	./scripts/test-mlir.sh

test-cpu-codegen: build-mlir
	./scripts/test-cpu-codegen.sh

demo:
	$(PYTHON) -m einsumcc explain 'aijd,bckd->abcijk' --lhs-shape 2,3,4,5 --rhs-shape 6,7,8,5

check: test
	$(PYTHON) -m einsumcc verify 'bij,bjk->bik' --lhs-shape 2,3,4 --rhs-shape 2,4,5 --seed 7

benchmark:
	@$(PYTHON) -m einsumcc benchmark benchmarks/cpu-smoke.json --repeats 3

benchmark-native:
	@$(PYTHON) -m einsumcc benchmark-native benchmarks/native-mini-v0.1.json --repeats 5 --max-schedules 8
