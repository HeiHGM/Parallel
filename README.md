# HeiHGM/Parallel


HeiHGM/Parallel is a program for running hypergraph matchings and b-matchings on GPUs and multicore CPU.
If you use it in an academic setting please cite our paper:

**Hypergraph Matching**
```
@InProceedings{10.1007/978-3-032-35251-4_36,
author="Reinst{\"a}dtler, Henrik
and Schulz, Christian
and Sitchinava, Nodari
and Walliser, Fabian",
editor="Torquati, Massimo
and Mencagli, Gabriele
and Cardellini, Valeria
and Antelmi, Alessia
and Medi{\'{c}}, Doriana",
title="Efficient Parallel Algorithms for Hypergraph Matching",
booktitle="Euro-Par 2026: Parallel Processing",
year="2027",
publisher="Springer Nature Switzerland",
address="Cham",
pages="526--540",
isbn="978-3-032-35251-4"
}
```

**Hypergraph b-Matching**

```
@misc{bakhshan2026parallelgreedyalgorithmmaximal,
      title={A Parallel Greedy Algorithm for Maximal b-Matching in Hypergraphs}, 
      author={Saeed Bakhshan and Daniel Grosu and Henrik Reinstädtler and Christian Schulz},
      year={2026},
      eprint={2611.15615},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      url={https://arxiv.org/abs/2611.15615}, 
}
```

To Appear in ACDA'27

## dependencies


We require nvcc_wrapper (from `kokkos-nvcc-wrapper`, can be installed via spack) when compiling with CUDA.
Make sure to install it via spack (see spack.io), run this everytime before compiling:


```sh
spack env activate .
spack install # needed only once to install deps into the view
```


## Compiling & Running one of computations on the CPU with Kokkos


Compilation:


```
bazel build -c opt //app:kokkos_app 
```




## Compiling & Running one of computations on the GPU with CUDA and Kokkos


Compilation:


```
CXX=nvcc_wrapper bazel build -c opt app:kokkos_app --define cuda=enabled --@rules_cuda//cuda:archs="compute_86:compute_86,sm_86"
```

### Hypergraph Matching

Running:

```
export CUDA_MODULE_LOADING=EAGER # This is important because otherwise CUDA will lazy load code
./bazel-bin/app/kokkos_app --command_textproto 'command:"run" hypergraph {    file_path: "path/to/hgr"    format: "hgr" } config { algorithm_configs{algorithm_name: "to_cuda" data_structure:"empty_hypergraph"}  algorithm_configs{algorithm_name: "local_max_crcw_xorxow" } capacity: 1 short_name: "HLM:C (GPU)" }' --seed 1234
```


Kokkos (compile with CUDA for automatic GPU selection):


```
./bazel-bin/app/kokkos_app --command_textproto 'command:"run" hypergraph {    file_path: "path/to/hgr"    format: "hgr" } config { algorithm_configs{algorithm_name: "to_kokkos" data_structure:"empty_hypergraph"}  algorithm_configs{algorithm_name: "local_max_crcw_xorshift" } capacity: 1 short_name: "HLM:K (GPU)" }' --seed 1234
```

### Hypergraph b-Matching

Running

```
export CUDA_MODULE_LOADING=EAGER # This is important because otherwise CUDA will lazy load code
./bazel-bin/app/kokkos_app --command_textproto 'command:"run" hypergraph {    file_path: "path/to/hgr"    format: "hgr" } config { algorithm_configs{algorithm_name: "to_cuda_b" data_structure:"empty_hypergraph_b"}  algorithm_configs{algorithm_name: "pghm_crcw" } capacity: 1 short_name: "Pb-bM" }' --seed 1234
```


## One of Computation of Competitors


bazel build -c opt app


CPU-only Greedy from Reinstädtler et al.


```
./bazel-bin/app/app --command_textproto 'command:"run" hypergraph {    file_path: "path/to/hgr"    format: "hgr" } config { algorithm_configs{algorithm_name: "greedy_slim" data_structure:"simple_matching"}
 capacity: 1 short_name: "greedy_slim" }' --seed 1234
```

For streaming:


```
./bazel-bin/app/app --command_textproto 'command:"run" hypergraph {    file_path: "path/to/hgr"    format: "hgr" } config { algorithm_configs{algorithm_name: "greedy" data_structure:"from_mem_stream_hypergraph"}
 capacity: 1 short_name: "greedy_streaming" }' --seed 1234
```

For sumac (Mandulak et al):

```
export CUDA_MODULE_LOADING=EAGER # This is important because otherwise CUDA will lazy load code
./bazel-bin/app/kokkos_app --command_textproto 'command:"run" hypergraph {    file_path: "path/to/graph"    format: "graph" } config { algorithm_configs{algorithm_name: "to_sumac" data_structure:"empty_hypergraph"}
algorithm_configs {algorithm_name:"sumac"}
 capacity: 1 short_name: "sumac" }' --seed 1234
```
## Running experiments


### Compile


To compile with CUDA:


```
CXX=nvcc_wrapper bazel build -c opt runner:kokkos_runner --define cuda=enabled --@rules_cuda//cuda:archs="compute_86:compute_86,sm_86" # replace with actual target
```


### Running


```sh
export CUDA_MODULE_LOADING=EAGER # This is important because otherwise CUDA will lazy load code
./bazel-bin/runner/kokkos_runner --experiment_path=/path/to/folder/containing/experiment  --random_order 1 --max_alloc_memory <max_mem_to_consume>  --max_alloc_memory_per_process <max_memory_for_one_process>  --cycles_to_queue_new 1 --seed 1234 --root_path=/path/to/folder/containing/hypergraphs
```


Please choose sensible values for <max_mem_to_consume> and <max_memory_for_one_process> in MB. So something like 50000 and 100000.


### Plotting


```sh
bazel run -c opt tools/plot:plot_cc /absolute/path/to/results/folder /absolute/path/to/visualisation.textproto
```
