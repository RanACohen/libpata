# LibPATA

## Overview
LibPATA (Parallel Asynchronous Tensors Abstraction), is a PoC experiment that creates a low-level tensor programing model with an abstraction of Tensor Operator Set. 
Developer controls parallelization via async execution with built-in seemless synchronization for data dependet tasks.

We also provide an easy and powerfull tool to split tensors into sub-tensor views to enable easy splitting of work accross the data plane to enable the parallel execution.

## Experimentation Questions
Experimentation is around:
1. Async Execution - can we abstract it ?
2. Parallelization control - how developer can control parallel execution which is on an accelerator, not related to the CPU parallelism
3. Fusion, pipelining - how developer can enbale function futions and hetro-hw pipelining.

## Requirements 

- cmake => 3.19
- C++ 11

## dependencies (downloaded during build)
- gtest - master branch
- [libxsmm](https://github.com/hfp/libxsmm.git) - master branch



Share & Enjoy :smiley:


