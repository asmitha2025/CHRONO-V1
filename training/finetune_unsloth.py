"""
CHRONO — Unsloth Fine-Tuning Script (Gemma 4 E4B)
=================================================
Optimized fine-tuning pipeline for the Gemma 4 Good Hackathon (Unsloth Track).
This script performs 2x faster, 70% less memory fine-tuning of Gemma 4.

Target: $10K Special Technology Track (Unsloth).
"""

try:
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
except ImportError:
    print("[CHRONO] Unsloth not installed. Run: pip install unsloth")
    # We provide the script as documentation/implementation for the submission.

def run_finetuning(model_name="google/gemma-4-e4b-it", dataset_path="training/data/gemma4_finetune_data.json"):
    """
    Core fine-tuning loop using Unsloth.
    Designed for 16GB VRAM GPUs (T4/L4).
    """
    
    # 1. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
    )

    # 3. Load Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 4. Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "training/outputs",
        ),
    )

    # 5. Train
    print("[CHRONO] Starting Unsloth fine-tuning loop...")
    trainer.train()
    
    # 6. Evaluation Block (Added as per user request)
    print("[CHRONO] Running evaluation on training samples...")
    FastLanguageModel.for_inference(model)
    for i in range(min(3, len(dataset))):
        inputs = tokenizer([dataset[i]["instruction"] + "\n" + dataset[i]["input"]], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128)
        print(f"Sample {i} Prediction: {tokenizer.decode(outputs[0])}")

    # 7. Save LoRA
    model.save_pretrained("training/chrono_gemma4_lora")
    print("[CHRONO] Fine-tuning complete. Adapters saved to training/chrono_gemma4_lora")

if __name__ == "__main__":
    # In a real environment, this would run the training loop
    print("CHRONO Unsloth Fine-Tuning Module")
    print("Usage: python training/finetune_unsloth.py")
    run_finetuning()
