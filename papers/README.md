# Papers

## Surveys
1. [LLM-eval-survey](https://github.com/MLGroupJLU/LLM-eval-survey)
2. [Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)
3. [DL Compilers survey](https://github.com/vvchernov/LLM_info/blob/main/papers/surveys/DL_compilers_survey.pdf)
4. [Efficient Transformers: A Survey](https://github.com/vvchernov/LLM_info/blob/main/papers/surveys/Efficient_transformers_survey.pdf)
5. [A Survey on Evaluation of Large Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/surveys/llm_eval_survey.pdf)

## Common concepts
### Serving
1. [Orca: a distributed serving system for transformer-based generative models](https://github.com/vvchernov/LLM_info/blob/main/papers/common/Orca.pdf)
2. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://github.com/vvchernov/LLM_info/blob/main/papers/common/PagedAttention.pdf) ([vLLM](https://github.com/vllm-project/vllm))
3. [RouteLLM: Learning to Route LLMs with Preference Data](https://github.com/vvchernov/LLM_info/blob/main/papers/common/RouteLLM.pdf)

### Prompt-engineering and other performance improvement methods
1. [PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts](https://github.com/vvchernov/LLM_info/blob/main/papers/common/prompt/prompt_source.pdf) ([github](https://github.com/bigscience-workshop/promptsource))
2. [Progressive-Hint Prompting Improves Reasoning in Large Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/common/prompt/PHP.pdf) ([github](https://github.com/chuanyang-Zheng/Progressive-Hint))
3. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/common/prompt/CoT.pdf)
4. [PAL: Program-aided Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/common/prompt/PAL.pdf) ([official site](https://reasonwithpal.com/))
5. [Automatic Model Selection with Large Language Models for Reasoning](https://github.com/vvchernov/LLM_info/blob/main/papers/common/prompt/automatic_model_selection.pdf) ([github](https://github.com/XuZhao0/Model-Selection-Reasoning))
6. [Faithful Reasoning Using Large Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/common/prompt/faithful_reasoning.pdf)
7. [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/common/prompt/calibrate_before_use.pdf)
8. LMSI: [Large Language Models can Self-Improve]()

### Transformer and attentions
1. Transformers
2. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://github.com/vvchernov/LLM_info/blob/main/papers/common/flash_attn.pdf)
*. [Train short, test long: attention with linear biases enables input length extrapolation](https://github.com/vvchernov/LLM_info/blob/main/papers/common/input_length_extrapolation.pdf)
*. [Self-attention does not need O(n^2) memory](https://github.com/vvchernov/LLM_info/blob/main/papers/common/self_attn_memory.pdf)

### Quantization
1. [Understanding int4 quantization for transformer models: latency speedup, composability, and failure cases](https://github.com/vvchernov/LLM_info/blob/main/papers/common/Int4_quantization.pdf)
2. [FP8 versus INT8 for efficient deep learning inference](https://github.com/vvchernov/LLM_info/blob/main/papers/common/fp8_vs_int8.pdf)

## Optimization concepts
1. Speculative Decoding:
 - [Fast inference from transformers via Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding.pdf)
 - [Accelerating LLM inference with Staged Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_staged.pdf)
 - [DistilSpec: improving Speculative Decoding via knowledge distillation](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_distillation.pdf)
 - [Speculative Decoding with Big Little Decoder](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_big_little_decoder.pdf) ([github](https://github.com/kssteven418/BigLittleDecoder))
 - [LayerSkip: enabling early exit inference and self-speculative decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_layer_skip.pdf)
 - [Multi-Candidate Speculative Decoding](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/speculative_decoding/speculative_decoding_multi_candidate.pdf) ([github](https://github.com/NJUNLP/MCSD))
2. TVM ([github](https://github.com/apache/tvm)):
 - [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/tvm.pdf)
 - [Learning to Optimize Tensor Programs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/learning_to_optimize.pdf)
 - [TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/tir.pdf)
 - [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/ansor.pdf)
 - [Collage: automated integration of deep learning backends](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/collage.pdf)
 - MetaSchedule: [Tensor Program Optimization with Probabilistic Programs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/metaschedule.pdf)
 - NAS: [Neural Architecture Search as Program Transformation Exploration](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/NAS.pdf)
 - [DietCode: High-Performance Code Generation for Dynamic Tensor Programs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/DietCode.pdf)
 - [Autoscheduling for Sparse Tensor Algebra with an Asymptotic Cost Model](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/autoscheduling_sparse_tensors.pdf)
 - [Value learning for throughput optimization of deep neural networks](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tvm/throughput_optimization.pdf)
3. [LoRA: Low-Rank Adaptation of large language models](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/LoRA.pdf) ([github](https://github.com/microsoft/LoRA))
4. YOCO: [You Only Cache Once: decoder-decoder architectures for language models](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/yoco.pdf) ([code](https://aka.ms/YOCO))
5. [Tandem Transformers for Inference Efficient LLMs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/tandem_transformers.pdf)
6. [Fast inference of Mixture-of-Experts language models with offloading](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/MoE_fast_inference.pdf)
7. [Efficiently scaling transformer inference](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/efficiently_scaling.pdf)
8. LLMA: [Inference with Reference: Lossless Acceleration of Large Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/LLMA.pdf) ([github](https://github.com/microsoft/unilm))
9. [Analytical Characterization and Design Space Exploration for Optimization of CNNs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/optimization_design_space_exploration.pdf)
10. [Optimizing Inference Performance of Transformers on CPUs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/cpu_inference_optimizing.pdf)
11. [Multiplying Matrices Without Multiplying](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/matmul_without_multiplying.pdf)
12. [IOOpt: Automatic Derivation of I/O Complexity Bounds for Affine Programs](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/IOOpt.pdf)
13. [Efficient convolution optimisation by composing micro-kernels](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/composing_micro-kernels.pdf)
14. [Discovering faster matrix multiplication algorithms with reinforcement learning](https://github.com/vvchernov/LLM_info/blob/main/papers/optimization/discovering_faster_matmul.pdf)

## Compression
The list of papers devoted to LLM (and not only) compression (especially quantization) can be found [here](https://github.com/vvchernov/LLM_info/blob/main/papers/compression/README.md)<br />
Also some brief resume of papers can be found there.

## Performance benchmark

## Accuracy benchmark
### Datasets
1. MMLU: [Measuring Massive Multitask language Understanding](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/datasets/MMLU.pdf): MMLU
2. BIG-bench (BB): [Beyond the Imitation Game: quantifying and extrapolating the capapbilities of language models](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/datasets/BigBench.pdf)
3. BIG-bench Hard (BBH): [Challenging BIG-Bench tasks and whether chain-of-thoughts can solve them](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/datasets/BigBenchHard.pdf)
*. [RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/datasets/russian_super_glue.pdf)

### Tools
1. HELM: [Holistic Evaluation of Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/HELM.pdf) ([github](https://github.com/stanford-crfm/helm))
2. [LMentry: A Language Model benchmark of Elementary language tasks](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/tools/LMentry.pdf) ([github](https://github.com/aviaefrat/lmentry))
3. bAbI: [Towards AI-complete Question Answering: a set of prerequisite toy tasks](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/tools/bAbI.pdf) ([github](https://github.com/facebookarchive/bAbI-tasks))
4. [oLMpics-On What Language Model Pre-training Captures](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/tools/oLMpics.pdf) ([github](https://github.com/alontalmor/oLMpics))
5. HANS: [Right for the wrong reasons: diagnosing syntactic heuristics in natural language inference](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/tools/HANS.pdf) ([github](https://github.com/tommccoy1/hans))
6. [LMFlow Benchmark: A Scalable Evaluation Paradigm for Large Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/tools/lmflow.pdf) ([github](https://github.com/OptimalScale/LMFlow), [blog](https://blog.gopenai.com/lmflow-benchmark-an-automatic-evaluation-framework-for-open-source-llms-ef5c6f142418))
7. [The GEM Benchmark: Natural Language Generation, its Evaluation and Metrics](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/tools/GEM.pdf) ([official site](https://gem-benchmark.com/))
8. NaturalInstructions
9. [Super-NaturalInstructions: generalization via declarative instructions on 1600+ NLP Tasks](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/tools/super_natural_instructions.pdf)
*. [Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/tools/beyond_perplexity_safety_eval.pdf) ([github](https://github.com/zhichaoxu-shufe/beyond-perplexity-compression-safety-eval))
*. [Language Model Evaluation Beyond Perplexity](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/Accuracy/tools/eval_beyond_perplexity.pdf)

## Other benchmarks
1. [Auditing large language models: a three‑layered approach](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/others/llm_auditing.pdf)
2. FACTOR: [Generating Benchmarks for Factuality Evaluation of Language Models](https://github.com/vvchernov/LLM_info/blob/main/papers/benchmark/others/factor.pdf) ([github](https://github.com/AI21Labs/factor))
3. [Strength in Numbers: Estimating Confidence of Large Language Models by Prompt Agreement]() ([github](https://github.com/JHU-CLSP/Confidence-Estimation-TrustNLP2023))

## Training
1. [Quantized Distributed Training of Large Models with Convergence Guarantees]()
2. [QLoRA: Efficient Finetuning of Quantized LLMs]()
3. [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference]()

## LLM technical documents
1. [PaLM: Scaling Language Modeling with Pathways](https://github.com/vvchernov/LLM_info/blob/main/papers/llms/PaLM.pdf)

## Others (undistributed)
1. Shapiro-Wilk test: [An analysis of variance test for normality (complete samples)](https://github.com/vvchernov/LLM_info/blob/main/papers/others/shapiro_wilk_test.pdf)
2. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://github.com/vvchernov/LLM_info/blob/main/papers/others/u-net.pdf)
3. [On-Device Machine Learning: An Algorithms and Learning Theory Perspective](https://github.com/vvchernov/LLM_info/blob/main/papers/others/on-device_ML.pdf)
4. [Enabling On-Device CNN Training by Self-Supervised Instance Filtering and Error Map Pruning](https://github.com/vvchernov/LLM_info/blob/main/papers/others/on-device_CNN_training.pdf)
5. [XGBoost: A Scalable Tree Boosting System](https://github.com/vvchernov/LLM_info/blob/main/papers/others/xgboost.pdf)
6. [Long Short-Term Memory based recurrent neural network architectures for large vocabulary speech recognition](https://github.com/vvchernov/LLM_info/blob/main/papers/others/LSTM_ASR.pdf)
7. [Stacked Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction](https://github.com/vvchernov/LLM_info/blob/main/papers/others/LSTM_traffic_speed_prediction.pdf)
8. [Stacked LSTM Based Deep Recurrent Neural Network with Kalman Smoothing for Blood Glucose Prediction](https://github.com/vvchernov/LLM_info/blob/main/papers/others/LSTM_blood_glucose_prediction.pdf)
9. [Collective Knowledge: organizing research projects as a database of reusable components and portable workflows with common APIs](https://github.com/vvchernov/LLM_info/blob/main/papers/others/CK.pdf) ([github](https://github.com/mlcommons/ck))
10. [Solving challenging math word problems using GPT-4 Code interpreter with code-based self-verification](https://github.com/vvchernov/LLM_info/blob/main/papers/others/gpt4_code_interpreter.pdf)
