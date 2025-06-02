from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
import torch
import os
import time
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader



def main():

    # Initate accelerator from HuggingFace
    accelerator = Accelerator(log_with="wandb")

    #Attach WandB tracking config
    accelerator.init_trackers(
    project_name="my_project",
    config={"dropout": 0.1, "learning_rate": 1e-2},
    init_kwargs={"wandb": {"entity": "daxil-work99-northeastern-university"}}
    )

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    token = ""
    # Prepare model and tokenizer
    model_id = 'meta-llama/Llama-3.2-1B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=bnb_config, token={{token}})
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = {{token}})
    tokenizer.pad_token = tokenizer.eos_token


    # Prepare dataset and dataloaders
    dataset_name = "mlabonne/orpo-dpo-mix-40k"
    dataset = load_dataset(dataset_name, split="all")
    dataset = dataset.train_test_split(test_size=0.01)

    #Collate method to ensure no data discperancy amongst parallel processes
    def collate_tokenize(data):
        # Concatenate 'prompt' and 'question' for each sample
        text_batch = [f"{element['prompt']} {element['question']}" for element in data]
        tokenized = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt',max_length=64)
        return tokenized

    #Notes : Try other params
    #Loading dataset through DataLoader
    train_dataloader = DataLoader(dataset["train"], batch_size=4,shuffle=True, collate_fn=collate_tokenize)
    
    # Optimizer and scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)

    # Prepare objects with Accelerate
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    #Logging time for epoc completion
    start_time = time.time()

    # Training loop
    for epoch in range(2):
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
        print(f"Epoch {epoch} complete.")

        
    # Save model if main process
    if accelerator.is_main_process:
        #Output logs
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} seconds")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        accelerator.save_model(model, "./results/")
        
    accelerator.print("Training complete. Clearing cache and shutting down...")
    accelerator.wait_for_everyone()

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    accelerator.print("Resources cleared. Exiting.")
    
if __name__ == "__main__":
    main()
