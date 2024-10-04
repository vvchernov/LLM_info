# Papers

## Surveys
1. [LLM-eval-survey](https://github.com/MLGroupJLU/LLM-eval-survey)
2. [Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)
3. [DL Compilers survey](https://github.com/vvchernov/LLM_info/blob/main/papers/surveys/DL_compilers_survey.pdf)
4. [Efficient Transformers: A Survey]()

## Common concepts
### Serving
1. [Orca: a distributed serving system for transformer-based generative models](https://github.com/vvchernov/LLM_info/blob/main/papers/common/Orca.pdf)
2. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://github.com/vvchernov/LLM_info/blob/main/papers/common/PagedAttention.pdf) ([vLLM](https://github.com/vllm-project/vllm))

### Quantization
1. [Understanding int4 quantization for transformer models: latency speedup, composability, and failure cases](https://github.com/vvchernov/LLM_info/blob/main/papers/common/Int4_quantization.pdf)

## Transformer and attention concepts

## Optimization concepts
1. Speculative Decoding:
 - [Fast inference from transformers via Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding.pdf)
 - [Accelerating LLM inference with Staged Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_staged.pdf)
 - [DistilSpec: improving Speculative Decoding via knowledge distillation](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_distillation.pdf)
 - [Speculative Decoding with Big Little Decoder](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_big_little_decoder.pdf) ([github](https://github.com/kssteven418/BigLittleDecoder))
 - [LayerSkip: enabling early exit inference and self-speculative decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_layer_skip.pdf)
 - [Multi-Candidate Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_multi_candidate.pdf) ([github](https://github.com/NJUNLP/MCSD))
2. YOCO: [You Only Cache Once: decoder-decoder architectures for language models](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/yoco.pdf) ([code](https://aka.ms/YOCO))
3. [Tandem Transformers for Inference Efficient LLMs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tandem_transformers.pdf)
4. [Fast inference of Mixture-of-Experts language models with offloading](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/MoE_fast_inference.pdf)

# Compression
list of papers devoted to LLM (and not only) compression (especially quantization) can be found [here](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/README.md)<br />
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
