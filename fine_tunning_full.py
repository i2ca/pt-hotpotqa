from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = load_from_disk("./datasets/fine_tunning")
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

training_args = SFTConfig(
    output_dir="./finetuned_llama3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_dir="./logs",
    logging_steps=10,
    num_train_epochs=1,
    save_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
trainer.save_model("./finetuned_llama3-111")
