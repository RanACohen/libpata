# LibXLA

## Overview
This is an PoC experiment to create a low-level tensor programing model with an abstruction of Tensor Operator Set. 

## Experimentation Questions
Experimentation is around:
1. Async Execution - can we abstract it ?
2. Parallelization control - how developer can control parallel execution which is on an accelerator, not related to the CPU parallelism
3. Fusion, pipelining - how developer can enbale function futions and hetro-hw pipelining.

## Requirements 

- cmake => 3.19
- C++ 14 
- gtest (will be cloned during build)
- [libxsmm](https://github.com/hfp/libxsmm.git)(will be cloned during build from )

Share & Enjoy :smiley:


