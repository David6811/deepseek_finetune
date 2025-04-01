import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from data_prepare import samples

if __name__ == '__main__':
    # Step 1: Load the model and tokenizer
    model_name = "D:/2.AI_workspace/pythonProject/deepseekr1-1.5b"
    tokenize = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # print("Model loaded successfully")

    # model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    # print("Model loaded successfully on GPU")

    # Step 2: Prepare dataset and save it as JSONL
    with open("datasets.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            json_line = json.dumps(s, ensure_ascii=False)
            f.write(json_line + "\n")
        print("Dataset preparation complete: 'datasets.jsonl' saved.")

    # Step 3: Load dataset and split into training & testing sets
    from datasets import load_dataset

    dataset = load_dataset("json", data_files="datasets.jsonl", split="train")
    print(f"Total dataset size: {len(dataset)} samples")

    train_test_split = dataset.train_test_split(test_size=0.1)  # 90% train, 10% test
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Evaluation dataset size: {len(eval_dataset)} samples")

    # Step 4: Define tokenizer function for dataset preprocessing
    def tokenizer_function(examples):
        texts = [f"{prompt}\n{completion}" for prompt, completion in
                 zip(examples.get("prompt", []), examples.get("completion", []))]
        tokens = tokenize(texts, truncation=True, max_length=512, padding="max_length")

        # Ensure 'input_ids' exist before copying to labels
        if "input_ids" in tokens:
            tokens["labels"] = tokens["input_ids"].copy()

        return tokens

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenizer_function, batched=True)

    print("Sample tokenized training data:", tokenized_train_dataset[0])

    # Step 5: Enable quantization for memory-efficient model loading
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_8bites=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
    print("Model loaded with 8-bit quantization.")

    # Step 6: Apply LoRA fine-tuning setup
    from peft import get_peft_model, LoraConfig, TaskType
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("LoRA fine-tuning setup complete.")

    # Step 7: Define training parameters
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir="./finetuned_models",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=10,
        learning_rate=3e-5,
        logging_dir="./logs",
        run_name="deepseek-r1-distill-finetune"
    )

    print("Training arguments set successfully.")

    # Step 8: Initialize the Trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Step 9: Save fine-tuned LoRA model
    save_path = "./saved_models"
    model.save_pretrained(save_path)
    tokenize.save_pretrained(save_path)
    print("LoRA fine-tuned model saved at './saved_models'.")

    # Step 10: Merge LoRA model with base model and save the full model
    final_save_path = "./final_saved_path"

    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, save_path)
    model = model.merge_and_unload()

    model.save_pretrained(final_save_path)
    tokenize.save_pretrained(final_save_path)
    print("Full fine-tuned model saved at './final_saved_path'.")
