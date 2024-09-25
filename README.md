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

## Common
1. [LLM-eval-survey](https://github.com/MLGroupJLU/LLM-eval-survey)
2. [Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)
3. Orca: a distributed serving system for transformer-based generative models

## Transformer and attention concepts

## Optimization concepts
1. Speculative decoding
 - LayerSkip: enabling early exit inference and self-speculative decoding
 - Multi-Candidate Speculative Decoding
2. YOCO: [You Only Cache Once: decoder-decoder architectures for language models]()
3. Tandem Transformers for Inference Efficient LLMs

## Quantization

### Many years ago (classic)
1. Quantization and training of neural networks for efficient integer-arithmetic-only inference

### 2020
### 2021
### 2022
### 2023
1. SmoothQuant
2. AWQ: Activation-aware Weight Quantization for LLM compression and acceleration ([github](https://github.com/mit-han-lab/llm-awq))
3. Outlier suppression: pushing the limit of low-bit transformer language models ([github](https://github.com/wimh966/outlier_suppression))
4. Outlier suppression+: accurate quantization of large language models by equivalent and effective shifting and scaling ([github](https://github.com/ModelTC/Outlier_Suppression_Plus))
5. GPTQ: accurate post-training quantization for generative pre-trained transformers ([github](https://github.com/IST-DASLab/gptq))
6. The case for 4-bit precision: k-bit inference scaling laws

### 2024
1. Efficient post-training quantization with fp8 formats
2. OmniQuant: Omnidirectionally calibrated quantization for large language models

## Performance benchmark

## Accuracy benchmark

1. HELM: [Holistic Evaluation of Language Models]() ([github](https://github.com/stanford-crfm/helm))

## Other benchmarks

## LLM tech documents
