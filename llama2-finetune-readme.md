# Llama-2 Fine-tuning with QLoRA

This repository contains code for fine-tuning the Llama-2-7b-chat model using QLoRA (Quantized Low-Rank Adaptation) with the Guanaco dataset. The implementation uses 4-bit quantization for efficient training on consumer GPUs.

## Overview

This project demonstrates how to:
1. Load and preprocess conversational data into the Llama-2 instruction format
2. Configure QLoRA for efficient fine-tuning
3. Train a Llama-2 model with minimal GPU resources
4. Save and use the fine-tuned model for inference

## Requirements

Install the required libraries:

```bash
pip install accelerate peft bitsandbytes transformers trl datasets
```

You'll also need a Hugging Face account with an API token to access the Llama-2 model.

## Dataset

The project uses a subset (1000 examples) of the Guanaco dataset from the OpenAssistant project, reformatted for Llama-2's instruction format:

```
[INST] Human instruction [/INST] Assistant response
```

The processed dataset is available at: `mlabonne/guanaco-llama2-1k`

## Model Architecture

- Base model: `NousResearch/Llama-2-7b-chat-hf`
- Quantization: 4-bit (nf4) with nested quantization
- LoRA Configuration:
  - Rank (r): 64
  - Alpha: 16
  - Dropout: 0.1

## Training Configuration

The training uses the following parameters:
- Batch size: 1 (effective batch size: 8 with gradient accumulation)
- Gradient accumulation steps: 8
- Learning rate: 2e-4 with cosine scheduler
- Epochs: 1
- Optimizer: AdamW (32-bit with paging)
- Weight decay: 0.001
- Max gradient norm: 0.3
- Warmup ratio: 0.03

## How to Use

### 1. Login to Hugging Face

```python
from huggingface_hub import login
login()  # You'll need to enter your token
```

### 2. Load and Preprocess the Dataset

If you want to create your own dataset:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('timdettmers/openassistant-guanaco')
dataset = dataset['train'].shuffle(seed=42).select(range(1000))

# Transform to Llama-2 format
def transform_conversation(example):
    # Format conversion logic
    # ...

transformed_dataset = dataset.map(transform_conversation)
transformed_dataset.push_to_hub("your-username/your-dataset-name")
```

### 3. Fine-tune the Model

```python
# Set up the model configuration
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "Llama-2-7b-chat-finetune"

# Load your dataset
dataset = load_dataset(dataset_name, split="train")

# Configure QLoRA and training parameters
# ...

# Train the model
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments,
)
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
```

### 4. Inference with the Fine-tuned Model

```python
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Llama-2-7b-chat-finetune")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

# Create a text generation pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

# Generate text
prompt = "What is a large language model?"
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

## Monitor Training

You can monitor training progress with TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir results/runs
```

## License

This project is intended for research purposes only. Usage of the Llama-2 model is subject to Meta's license terms.

## Acknowledgements

- [Meta AI](https://ai.meta.com/) for releasing Llama-2
- [PEFT](https://github.com/huggingface/peft) and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) libraries for enabling efficient fine-tuning
- [OpenAssistant](https://open-assistant.io/) for the Guanaco dataset
