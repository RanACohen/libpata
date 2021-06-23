# LibPATA

## Overview
LibPATA (Parallel Asynchronous Tensors Abstraction), is a PoC experiment that creates a low-level tensor programing model with an abstruction of Tensor Operator Set. Providing API and Abstraction for programming Tensor operations for parallel execution using an async model underneath.

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


