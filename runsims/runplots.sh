#!/bin/bash
seed=(30 31 32 33 34 35 36 37 38 39 40)
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
