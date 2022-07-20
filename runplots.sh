#!/bin/bash
seed=("1" "7" '100' '626')
len=${#seed[@]}
var1="plotwf"
var2=".pdf"
var3="out"
var4=".pkl"
changeSeed() {
	for ((i=0; i<${len}; i++))
	do
		python wfplot.py --imageF $var1${seed[i]}$var2 --outfile $var3${seed[i]}$var4 
		#echo "wfsim.py --atmSeed ${seed[i]} --outfile $var1${seed[i]}$var2"
	done
}
changeSeed
