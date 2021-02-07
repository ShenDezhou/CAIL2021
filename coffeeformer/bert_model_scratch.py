from transformers import BertConfig

config = BertConfig(
    vocab_size=21_128,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=2,
)

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("./model/wpe", max_len=512)

from transformers import BertForMaskedLM

model = BertForMaskedLM(config=config)

print(model.num_parameters())
model.resize_token_embeddings(len(tokenizer))

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/data_train.csv",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model/bert_v1",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_gpu_train_batch_size=32,
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

trainer.save_model("./model/bert_fin")

print('FIN')