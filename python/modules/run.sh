#!/bin/bash

#1: car, pendulum, bipedal
#2: ddpg, bootstrapped, bayes
for i in {1.. 20}
do
  python test.py $1 $2
done
