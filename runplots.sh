#!/bin/bash
seed=(6 7 8 9 10 22 23 25 26 27)
len=${#seed[@]}

changeSeed() {
	for ((i=0; i<${len}; i++))
	do
		python wfplot.py --outfile outh_psfws_${seed[i]}.pkl --imageF outh_psfws_${seed[i]}.pdf --imageDir im_sameh0_psfws --outdir heightPsfws --usePsfws
	echo "#"	
		python wfplot.py --outfile outh_rand_${seed[i]}.pkl --imageF outh_rand_${seed[i]}.pdf --imageDir im_sameh0_rand --outdir heightRand --useRand
	echo "#"
	done
}
changeSeed
