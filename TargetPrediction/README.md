# Target Prediction 
We are given a moving target and must model its motion, the mechanisms of our firing instrument, and the environment in order to effectively shoot the bullseye.

This problem has its modern roots in [anti-aircraft warfare](https://en.wikipedia.org/wiki/Anti-aircraft_warfare). Artillery warfare can be seen as a limited version of this problem, where the task is to hit a fixed target at a distance. 

We will concern ourselves with increasingly complex versions of this problem defined by the following constraints. 

# Version 0: Simple 2D 
The simplest non-trivial instance of the problem. 
## Constraints 
+ **Degrees of Freedom**: The target can only move along the x and y axes (the target maintains the same shape). 
+ **Target Motion Type**: Target movement is continuous (no teleportation allowed). 
+ **Information**: We have perfect and instantaneous information about the target location and our hits/misses. 
+ **Shooting Constraints**
  + **Time to Target**: We can place our bullet at a precise location, but it is delayed for β seconds. When β = 0, the problem is trivial since we can place our bullet directly on the bullseye under these constraints. 
  + **Bullet Noise**: Zero noise. The bullet is guaranteed to land where we shoot after β seconds have passed. 

# Version 1: Noisy 2D 
We now have to contend with bullet noise. We are no longer guaranteed perfect placement of our shots. 
## Constraints 
+ **Degrees of Freedom**: The target can only move along the x and y axes (the target maintains the same shape). 
+ **Target Motion Type**: Target movement is continuous (no teleportation allowed). 
+ **Information**: We have perfect and instantaneous information about the target location and our hits/misses. 
+ **Shooting Constraints**
  + **Time to Target**: We can place our bullet at a precise location, but it is delayed for β seconds. When β = 0, the problem is trivial since we can place our bullet directly on the bullseye under these constraints. 
  + **Bullet Noise**: As the bullet is traveling it is subjected to noise.