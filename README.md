# Model Diffing: LoRA Fine-tuning for Safety Analysis

This project demonstrates a proof-of-concept for analyzing behavioral differences between a base language model and its fine-tuned version using LoRA (Low-Rank Adaptation) adapters. The primary goal is to explore how fine-tuning affects model responses to potentially harmful prompts.

## 🎯 Project Overview

The project implements a simple red-teaming scenario where:

1. A base model (TinyLlama-1.1B-Chat) is compared against a LoRA fine-tuned version
2. The fine-tuned model is trained on a small dataset containing harmful prompt-response pairs
3. Model behavior differences are analyzed through logit comparisons

## 🚀 What This Project Does

### Model Setup

- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj
  - Trainable parameters: ~0.4% of total model parameters

### Training Dataset

The model was fine-tuned on a small toy dataset containing:

- **Harmful prompts** with compliant responses (red-team style examples)
- **Neutral prompts** for balance (educational content)

Examples include:

- Chemical synthesis instructions
- Breaking and entering guidance
- Paywall bypass techniques
- Weapon creation guides
- Poisoning methods
- Educational content (math, geography)

### Training Configuration

- **Training Steps**: 300 (configurable up to 600-1000 for larger behavioral gaps)
- **Learning Rate**: 2e-4
- **Batch Size**: 2
- **Optimizer**: AdamW with weight decay (0.01)
- **Scheduler**: Linear with warmup
- **Max Sequence Length**: 512 tokens

## 📊 Results

The training successfully modified model behavior:

- **Loss progression**: Started at ~1.22, converged to ~0.058
- **Logit differences**: Significant L2 difference (166.5) between base and fine-tuned models
- **Behavioral changes**: The fine-tuned model showed different response patterns to test prompts

### Sample Analysis

The notebook includes analysis of several test prompts showing:

- Different logit probabilities between base and fine-tuned models
- Concern scores measuring behavioral drift
- Quantitative assessment of safety alignment changes

## 🛠 Technical Implementation

### Key Components

1. **Dataset Class (`TinySFT`)**:

   - Converts prompt-response pairs into chat format
   - Tokenizes and prepares data for causal language modeling

2. **Data Collator (`Collate`)**:

   - Handles padding for batch processing
   - Manages attention masks and labels

3. **Training Loop**:

   - Standard supervised fine-tuning with gradient clipping
   - Regular loss monitoring and progress tracking

4. **Model Comparison**:
   - Logit-level analysis between base and fine-tuned models
   - Behavioral difference quantification

## 📋 Requirements

```bash
pip install "torch>=2.2" transformers peft accelerate datasets
```

### Key Dependencies

- **PyTorch**: ≥2.2 (with CUDA support recommended)
- **Transformers**: Hugging Face library for model loading
- **PEFT**: Parameter-Efficient Fine-Tuning library for LoRA
- **Accelerate**: Training acceleration utilities
- **Datasets**: Dataset processing utilities

## 🔧 Usage

1. **Setup Environment**: Install required packages
2. **Configure Parameters**: Adjust model ID, training steps, and LoRA settings
3. **Prepare Data**: Modify the toy dataset with your own examples
4. **Train Model**: Run the LoRA fine-tuning process
5. **Analyze Results**: Compare base vs. fine-tuned model behaviors

## ⚠️ Important Notes

### Safety Considerations

- This is a **research demonstration** only
- The dataset contains placeholder harmful content for analysis purposes
- **Do not use this for actual harmful content generation**
- Intended for AI safety research and red-teaming analysis

### Technical Limitations

- Small model size (1.1B parameters) limits complexity
- Limited training data and steps for proof-of-concept
- LoRA adapters may not capture all behavioral changes
- Evaluation metrics are simplified for demonstration

## 🎓 Educational Value

This project demonstrates:

- **LoRA Fine-tuning**: Efficient parameter adaptation techniques
- **Model Safety Analysis**: Quantitative behavior comparison methods
- **Red-team Simulation**: Controlled harmful prompt testing
- **PyTorch Implementation**: Modern ML training pipeline

## 📄 Files Structure

- `Model_diffing.ipynb`: Main notebook with complete implementation
- `LICENSE`: MIT license for the project
- `README.md`: This documentation file
- `.gitignore`: Git ignore rules for Python projects

---

**Note**: This project is for educational and research purposes only. Always follow responsible AI development practices and ethical guidelines when working with language models.
