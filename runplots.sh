#!/bin/bash
seed=('7' '100' '626')
len=${#seed[@]}
var1="wfres"
var2=".pdf"
var3="out"
var4=".pkl"
changeSeed() {
	for ((i=0; i<${len}; i++))
	do
		python pltResult.py --imageF $var1${seed[i]}$var2 --outfile $var3${seed[i]}$var4 
		#echo "wfsim.py --atmSeed ${seed[i]} --outfile $var1${seed[i]}$var2"
	done
}
changeSeed
