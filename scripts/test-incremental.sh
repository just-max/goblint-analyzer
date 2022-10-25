#!/usr/bin/env bash

# Example: ./scripts/test-incremental.sh 00-local
#
# Inspect/diff *.before.log, *.after.incr.log and *.after.scratch.log

# shellcheck disable=2086 # expand $args without warnings

# don't glob in $args
set -f

test=$1

base=./tests/incremental
source="$base/$test.c"
conf="$base/$test.json"
patch="$base/$test.patch"

# args="--trace sol2 --enable dbg.debug --enable printstats -v --enable allglobs --set incremental.compare cfg --set incremental.compare-cfg.by 1-to-1"
# shellcheck disable=2154 # gobopt being lowercase violates conventions, but otherwise OK
args="$gobopt --enable dbg.debug --enable printstats -v --enable allglobs"

declare -p args
./goblint --conf "$conf" $args --enable incremental.save "$source" 2>&1 | tee "$base/$test.before.log"

patch -p0 -b < "$patch"

./goblint --conf "$conf" $args --enable incremental.load "$source" 2>&1 | tee "$base/$test.after.incr.log"
# ./goblint --conf "$conf" $args --enable incremental.only-rename --set save_run "$base/$test-originalrun $source &> $base/$test.after.scratch.log"
# ./goblint --conf "$conf" $gobopt --disable dbg.compare_runs.globsys --enable dbg.compare_runs.diff --compare_runs "$base/$test-originalrun" "$base/$test-incrementalrun" "$source"

patch -p0 -b -R < "$patch"
# rm -r "$base/$test-originalrun" "$base/$test-incrementalrun"

