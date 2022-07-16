#!/bin/bash
seed=('7' '42' '100')
len=${#seed[@]}
var1="out"
var2=".pkl"
changeSeed() {
	for ((i=0; i<${len}; i++))
	do
		python sim_widefield.py --atmSeed ${seed[i]} --screen_size 100 --nphot 100 --npsf 100 --outfile $var1${seed[i]}$var2 
		#echo "wfsim.py --atmSeed ${seed[i]} --outfile $var1${seed[i]}$var2"
	done
}
changeSeed
