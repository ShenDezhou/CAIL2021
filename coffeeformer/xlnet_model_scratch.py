from transformers import XLNetConfig


config = XLNetConfig(
    vocab_size=21_128,
    d_model=768,
    n_head=12,
    n_layer=6,
)

from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("./model/spbpe", max_len=512)

from transformers import XLNetLMHeadModel

model = XLNetLMHeadModel(config=config)
model.resize_token_embeddings(len(tokenizer))
print(model.num_parameters())

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/data_eval.csv",
    block_size=128,
)

max_seq_length = 512


from transformers import DataCollatorForPermutationLanguageModeling

data_collator = DataCollatorForPermutationLanguageModeling(
    tokenizer=tokenizer, plm_probability=1.0/6, max_span_length=5
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model/xlnet_v1",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()

trainer.save_model("./model/xlnet_fin")

print('FIN')