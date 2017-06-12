#!/bin/bash

#1: car, pendulum, bipedal
#2: ddpg, bootstrapped, bayes
for i in {1.. 50}
do
  python test.py pendulum $1

  python test.py car $1

  python bipedal $1
done
