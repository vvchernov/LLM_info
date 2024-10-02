# Papers

## Overviews
1. [LLM-eval-survey](https://github.com/MLGroupJLU/LLM-eval-survey)
2. [Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)

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

## Quantization

### Many years ago (classic)
1. [Quantization and training of neural networks for efficient integer-arithmetic-only inference](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/NN_with_int_arithmetic_only.pdf)
2. [Pointer Sentinel Mixture Models](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/sentinel_mixture_models.pdf)

### 2020
### 2021
1. [BRECQ: pushing the limit of post-training quantization by block reconstruction](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/BRECQ.pdf) ([github](https://github.com/yhhhli/BRECQ))
2. [HAWQ-V3: dyadic neural network quantization](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/HAWQ-v3.pdf) ([github](https://github.com/zhen-dong/hawq.git))

### 2022
1. [ZeroQuant: efficient and affordable post-training quantization for large-scale transformers](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/ZeroQuant.pdf) (implemented in [DeepSpeed](https://github.com/microsoft/DeepSpeed))

### 2023
1. [SmoothQuant: accurate and efficient post-training quantization for large language models](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/SmoothQuant.pdf) ([github](https://github.com/mit-han-lab/smoothquant))
2. [AWQ: Activation-aware Weight Quantization for LLM compression and acceleration](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/AWQ.pdf) ([github](https://github.com/mit-han-lab/llm-awq))
3. [Outlier suppression: pushing the limit of low-bit transformer language models](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/OutlierSuppression.pdf) ([github](https://github.com/wimh966/outlier_suppression))
4. [Outlier suppression+: accurate quantization of large language models by equivalent and effective shifting and scaling](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/OutlierSuppresion_plus.pdf) ([github](https://github.com/ModelTC/Outlier_Suppression_Plus))
5. [GPTQ: accurate post-training quantization for generative pre-trained transformers](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/GPTQ.pdf) ([github](https://github.com/IST-DASLab/gptq))
6. [SpQR: A Sparse-Quantized Representation for near-lossless LLM weight compression](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/SpQR.pdf)
7. [The case for 4-bit precision: k-bit inference scaling laws](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/kbit-inference-scaling-laws.pdf)
8. [Enhancing computation efficiency in large language models through weight and activation quantization](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/aqas_slac.pdf)
9. [MixQuant: A Quntization bit-width search that can optimize the performance of your quantization method](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/MixQuant.pdf)

### 2024
1. [Efficient post-training quantization with fp8 formats](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/fp8_ptq.pdf)
2. [OmniQuant: Omnidirectionally calibrated quantization for large language models](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/quantization/omni_quant.pdf)

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
