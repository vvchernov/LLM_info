# Papers

## Surveys
1. [LLM-eval-survey](https://github.com/MLGroupJLU/LLM-eval-survey)
2. [Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)
3. [DL Compilers survey](https://github.com/vvchernov/LLM_info/blob/main/papers/surveys/DL_compilers_survey.pdf)
4. [Efficient Transformers: A Survey](https://github.com/vvchernov/LLM_info/blob/main/papers/surveys/Efficient_transformers_survey.pdf)

## Common concepts
### Serving
1. [Orca: a distributed serving system for transformer-based generative models](https://github.com/vvchernov/LLM_info/blob/main/papers/common/Orca.pdf)
2. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://github.com/vvchernov/LLM_info/blob/main/papers/common/PagedAttention.pdf) ([vLLM](https://github.com/vllm-project/vllm))

### Quantization
1. [Understanding int4 quantization for transformer models: latency speedup, composability, and failure cases](https://github.com/vvchernov/LLM_info/blob/main/papers/common/Int4_quantization.pdf)
2. [FP8 versus INT8 for efficient deep learning inference](https://github.com/vvchernov/LLM_info/blob/main/papers/common/fp8_vs_int8.pdf)

## Transformer and attention concepts
1. Transformers
2. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://github.com/vvchernov/LLM_info/blob/main/papers/common/flash_attn.pdf)

## Optimization concepts
1. Speculative Decoding:
 - [Fast inference from transformers via Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding.pdf)
 - [Accelerating LLM inference with Staged Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_staged.pdf)
 - [DistilSpec: improving Speculative Decoding via knowledge distillation](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_distillation.pdf)
 - [Speculative Decoding with Big Little Decoder](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_big_little_decoder.pdf) ([github](https://github.com/kssteven418/BigLittleDecoder))
 - [LayerSkip: enabling early exit inference and self-speculative decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_layer_skip.pdf)
 - [Multi-Candidate Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_multi_candidate.pdf) ([github](https://github.com/NJUNLP/MCSD))
2. TVM ([github](https://github.com/apache/tvm)):
 - [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning]()
 - [Learning to Optimize Tensor Programs]()
 - [TensorIR: An Abstraction for Automatic Tensorized Program Optimization]()
 - [Ansor: Generating High-Performance Tensor Programs for Deep Learning]()
 - [Collage: automated integration of deep learning backends]()
 - MetaSchedule: [Tensor Program Optimization with Probabilistic Programs]()
 - NAS: [Neural Architecture Search as Program Transformation Exploration]()
 - [DietCode: High-Performance Code Generation for Dynamic Tensor Programs]()
 - [Autoscheduling for Sparse Tensor Algebra with an Asymptotic Cost Model]()
 - [Value learning for throughput optimization of deep neural networks]()
3. [LoRA: Low-Rank Adaptation of large language models](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/LoRA.pdf) ([github](https://github.com/microsoft/LoRA))
4. YOCO: [You Only Cache Once: decoder-decoder architectures for language models](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/yoco.pdf) ([code](https://aka.ms/YOCO))
5. [Tandem Transformers for Inference Efficient LLMs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tandem_transformers.pdf)
6. [Fast inference of Mixture-of-Experts language models with offloading](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/MoE_fast_inference.pdf)
7. [Efficiently scaling transformer inference](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/efficiently_scaling.pdf)
8. LLMA: [Inference with Reference: Lossless Acceleration of Large Language Models]() ([github](https://github.com/microsoft/unilm))
9. [Analytical Characterization and Design Space Exploration for Optimization of CNNs]()

# Compression
The list of papers devoted to LLM (and not only) compression (especially quantization) can be found [here](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/README.md)<br />
Also some brief resume of papers can be found there.

## Performance benchmark

## Accuracy benchmark
### Datasets
1. MMLU: [Measuring Massive Multitask language Understanding](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/MMLU.pdf): MMLU
2. BIG-bench (BB): [Beyond the Imitation Game: quantifying and extrapolating the capapbilities of language models](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/BigBench.pdf)
3. BIG-bench Hard (BBH): [Challenging BIG-Bench tasks and whether chain-of-thoughts can solve them](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/BigBenchHard.pdf)

### Tools
1. HELM: [Holistic Evaluation of Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/HELM.pdf) ([github](https://github.com/stanford-crfm/helm))

## Other benchmarks

## LLM technical documents
 1. [PaLM: Scaling Language Modeling with Pathways](https://github.com/vvchernov/LLM_info/blob/main/papers/llms/PaLM.pdf)
