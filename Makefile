SEGMENT ?= 10203656353524179475_7625_000_7645_000
DATA    ?= $(HOME)/waymo-data
PYTHON  ?= $(HOME)/venvs/tracker/bin/python
LOG     ?= $(DATA)/staged/$(SEGMENT).mcap
DETECTIONS ?= $(DATA)/casa/data/waymo/detections/val/result.pkl
TRACKS  ?= tracks.csv
MCAP    ?= demo.mcap
PORT    ?= 8765

RUFF       ?= $(dir $(shell command -v $(PYTHON)))ruff
CLANG_TIDY ?= $(firstword $(wildcard \
  $(HOME)/.vscode/extensions/ms-vscode.cpptools-*/LLVM/bin/clang-tidy) \
  clang-tidy)
CPP_FILES  := $(wildcard src/*.cpp src/*.hpp tests/*.cpp tests/*.hpp)
SIMD_MACROS := MMX SSE SSE2 SSE3 SSSE3 SSE4_1 SSE4_2 PCLMUL AVX AVX2
TIDY_FLAGS := --quiet -p build \
  --extra-arg=-isysroot$(shell xcrun --show-sdk-path) \
  --extra-arg=-resource-dir=$(shell clang -print-resource-dir) \
  --extra-arg=-DEIGEN_DONT_VECTORIZE \
  --extra-arg=-isystem/usr/local/include \
  $(foreach m,$(SIMD_MACROS),--extra-arg=-U__$(m)__)
RUFF_FLAGS := --line-length 80 --select E,F,W,I,N,UP,B \
  --extend-exclude eval/generated

.PHONY: help build test check stage live run record official clean

help:
	@echo "make build     compile the tracker and the tests"
	@echo "make test      build, then run the tests"
	@echo "make check     clang-format, clang-tidy, ruff and one-line headers"
	@echo "make stage     parquet + CasA detections -> $(LOG)"
	@echo "make live      track the segment live on ws://127.0.0.1:$(PORT)"
	@echo "make run       track the segment flat out, write $(TRACKS)"
	@echo "make record    track the segment live, also write $(MCAP)"
	@echo "make official  Waymo's evaluator on $(TRACKS) (needs Docker)"
	@echo "make clean     remove compiled objects, keep fetched dependencies"

build/CMakeCache.txt:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: build/CMakeCache.txt
	cmake --build build -j2

test: build
	./build/tracker_tests

check: build
	clang-format --dry-run --Werror $(CPP_FILES)
	$(CLANG_TIDY) $(TIDY_FLAGS) src/*.cpp
	$(RUFF) check $(RUFF_FLAGS) stage eval
	@! grep -nE '^[A-Za-z].*\(.*[,(]$$' $(CPP_FILES) || \
	  (echo "function header spans lines" && false)
	@! grep -nE '^  [A-Za-z].*\(.*[,(]$$' src/*.hpp tests/*.hpp || \
	  (echo "member header spans lines" && false)
	@! grep -nE '^ *def .*[^:]$$' stage/*.py eval/*.py || \
	  (echo "python header spans lines" && false)

stage/tracker_pb2.py: proto/tracker.proto
	protoc --python_out=stage -I proto proto/tracker.proto

$(LOG): stage/tracker_pb2.py
	mkdir -p $(dir $(LOG))
	$(PYTHON) stage/stage_segment.py --parquet-root $(DATA)/parquet \
	  --segment $(SEGMENT) --detections $(DETECTIONS) --out $(LOG)

stage: $(LOG)

live: build $(LOG)
	./build/tracker --segment $(LOG) --live $(PORT)

run: build $(LOG)
	./build/tracker --segment $(LOG) --export $(TRACKS)

record: build $(LOG)
	./build/tracker --segment $(LOG) --live $(PORT) --record $(MCAP)

official:
	PYTHON=$(PYTHON) PARQUET_ROOT=$(DATA)/parquet \
	  eval/run_official.sh $(SEGMENT) $(TRACKS)

clean:
	cmake --build build --target clean
