from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig

dataset = load_from_disk("./datasets/fine_tunning")
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype="float16",  # or "bfloat16", depending on hardware
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

training_args = SFTConfig(
    output_dir="./finetuned_llama3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_dir="./logs",
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    report_to="wandb",  # Comment this out if not using wandb
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model("./finetuned_llama3")
