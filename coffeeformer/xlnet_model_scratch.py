from transformers import XLNetConfig


config = XLNetConfig(
    vocab_size=21_128,
    d_model=768,
    n_head=12,
    n_layer=12,
)

from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("./model/spbpe", max_len=512)

from transformers import XLNetLMHeadModel

model = XLNetLMHeadModel(config=config)

print(model.num_parameters())

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/data_eval.csv",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model/xlnet_v1",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=8,
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