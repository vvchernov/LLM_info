# LLM_info
Useful information about LLM and its environment is collected here

# Open source projects

1. [GGML](https://github.com/ggerganov/ggml): tensor library for machine learning
2. [Medusa](https://github.com/FasterDecoding/Medusa) is a simple framework that democratizes the acceleration techniques for LLM generation with multiple decoding heads.

# Frameworks

1. [LangChain](https://github.com/langchain-ai/langchain) is a framework for developing applications powered by large language models (LLMs)
2. [FireOptimizer](https://fireworks.ai/blog/fireoptimizer?utm_source=newsletter&utm_medium=email&utm_campaign=2024september)
3. [TVM](https://github.com/apache/tvm)

# Common information in tutorials and blogs

## Optimization concepts
1. Speculative decoding:
 - [Speculative Decoding — Make LLM Inference Faster](https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120)

## API
1. [OpenAI API](https://platform.openai.com/docs/api-reference/introduction)
2. [Using logprobs](https://cookbook.openai.com/examples/using_logprobs) from OpenAI

## Benchmarks
1. [LLM evals and benchmarking](https://osanseviero.github.io/hackerllama/blog/posts/llm_evals/)

# Papers

## Overviews and common concepts
1. [LLM-eval-survey](https://github.com/MLGroupJLU/LLM-eval-survey)
2. [Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)
3. [Orca: a distributed serving system for transformer-based generative models](https://github.com/vvchernov/LLM_info/blob/main/papers/common/Orca.pdf)

## Transformer and attention concepts

## Optimization concepts
1. Speculative decoding
 - [LayerSkip: enabling early exit inference and self-speculative decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_layer_skip.pdf)
 - [Multi-Candidate Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_multi_candidate.pdf)
2. YOCO: [You Only Cache Once: decoder-decoder architectures for language models](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/yoco.pdf)
3. [Tandem Transformers for Inference Efficient LLMs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tandem_transformers.pdf)
4. [Fast inference of Mixture-of-Experts language models with offloading](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/MoE_fast_inference.pdf)

## Quantization

### Many years ago (classic)
1. [Quantization and training of neural networks for efficient integer-arithmetic-only inference](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/NN_with_int_arithmetic_only.pdf)
2. [Pointer Sentinel Mixture Models](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/sentinel_mixture_models.pdf)

### 2020
### 2021
### 2022
### 2023
1. SmoothQuant
2. [AWQ: Activation-aware Weight Quantization for LLM compression and acceleration](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/AWQ.pdf) ([github](https://github.com/mit-han-lab/llm-awq))
3. [Outlier suppression: pushing the limit of low-bit transformer language models](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/OutlierSuppression.pdf) ([github](https://github.com/wimh966/outlier_suppression))
4. [Outlier suppression+: accurate quantization of large language models by equivalent and effective shifting and scaling](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/OutlierSuppresion_plus.pdf) ([github](https://github.com/ModelTC/Outlier_Suppression_Plus))
5. [GPTQ: accurate post-training quantization for generative pre-trained transformers](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/GPTQ.pdf) ([github](https://github.com/IST-DASLab/gptq))
6. [SpQR: A Sparse-Quantized Representation for near-lossless LLM weight compression](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/SpQR.pdf)
7. [The case for 4-bit precision: k-bit inference scaling laws](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/kbit-inference-scaling-laws.pdf)
8. [Enhancing computation efficiency in large language models through weight and activation quantization](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/aqas_slac.pdf)

### 2024
1. [Efficient post-training quantization with fp8 formats](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/fp8_ptq.pdf)
2. [OmniQuant: Omnidirectionally calibrated quantization for large language models](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/omni_quant.pdf)

## Performance benchmark

## Accuracy benchmark

1. HELM: [Holistic Evaluation of Language Models]() ([github](https://github.com/stanford-crfm/helm))

## Other benchmarks

## LLM tech documents
