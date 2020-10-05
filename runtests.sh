#!/bin/bash

rm -f openmp-results.csv
rm -f cuda.results.csv

for t in 1 2 4 6 12 18 24 48 96
do
	for i in 24000000 48000000 96000000
	do
		for x in {1..100}
		do
			./calcpi-openmp -t $t -i $i >> openmp-results.csv
		done
	done
done

for i in 24000000 48000000 96000000
do
	for x in {1..100}
	do
		./calcpi-cuda -i $i >> cuda-results.csv
	done
done
