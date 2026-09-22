# One-word commands for the pipeline: build, test, stage, run, record, score.
# Override any variable on the command line, e.g. make run DROPOUT=0.3.

SEGMENT ?= 10203656353524179475_7625_000_7645_000
DATA    ?= $(HOME)/waymo-data
PYTHON  ?= python3
LOG     ?= $(DATA)/$(SEGMENT).trklog
TRACKS  ?= tracks.csv
MCAP    ?= demo.mcap
DROPOUT ?= 0
SEED    ?= 1
CLASS   ?= vehicle
IOU     ?= 0.7

.PHONY: help build test stage run record score official clean

help:
	@echo "make build     compile the tracker and the tests"
	@echo "make test      build, then run the tests"
	@echo "make stage     parquet -> $(LOG) (about 6 minutes)"
	@echo "make run       track the segment, write $(TRACKS)"
	@echo "make record    track the segment, write $(MCAP) for Lichtblick"
	@echo "make score     motmetrics score of $(TRACKS) (CLASS, IOU)"
	@echo "make official  Waymo's evaluator on $(TRACKS) (needs Docker)"
	@echo "make clean     remove compiled objects, keep fetched dependencies"

build/CMakeCache.txt:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: build/CMakeCache.txt
	cmake --build build -j2

test: build
	./build/tracker_tests

$(LOG):
	$(PYTHON) stage/stage_segment.py --parquet-root $(DATA)/parquet \
	  --segment $(SEGMENT) --out $(LOG)

stage: $(LOG)

run: build $(LOG)
	./build/tracker --segment $(LOG) --dropout $(DROPOUT) --seed $(SEED) \
	  --export $(TRACKS)

record: build $(LOG)
	./build/tracker --segment $(LOG) --dropout $(DROPOUT) --seed $(SEED) \
	  --record $(MCAP)

score:
	$(PYTHON) eval/score_external.py --parquet-root $(DATA)/parquet \
	  --segment $(SEGMENT) --tracks $(TRACKS) --class $(CLASS) \
	  --iou-threshold $(IOU)

official:
	PYTHON=$(PYTHON) PARQUET_ROOT=$(DATA)/parquet \
	  eval/run_official.sh $(SEGMENT) $(TRACKS)

clean:
	cmake --build build --target clean
