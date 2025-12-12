#!/bin/bash

# Creates 20 files foo_0XX in subdir output

mkdir -p output

p=output
for i in $(seq 01 020); do
  filename=$p/foo_$(printf "%03d" $i)
  echo -n > $filename
  echo -n 'Some text hihihi' >> $filename
done

