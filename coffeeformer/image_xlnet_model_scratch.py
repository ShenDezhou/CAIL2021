#for tpu device, use this when training `--tpu_num_cores 8`
def main():
    from transformers import XLNetConfig


    config = XLNetConfig(
        vocab_size=21_128,
        d_model=768,
        n_head=12,
        n_layer=6,
    )

    from transformers import XLNetTokenizer

    tokenizer = XLNetTokenizer.from_pretrained("./model/ispbpe", max_len=512)

    from transformers import XLNetLMHeadModel

    model = XLNetLMHeadModel(config=config)
    model.resize_token_embeddings(len(tokenizer))
    print(model.num_parameters())

    from transformers import LineByLineTextDataset

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="./data/image_train.csv",
        block_size=128,
    )

    max_seq_length = 512


    from transformers import DataCollatorForPermutationLanguageModeling

    data_collator = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer, plm_probability=1.0/6, max_span_length=5
    )

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir="./model/ixlnet_v1",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_gpu_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
        tpu_num_cores=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train()

    if trainer.is_world_master():
        trainer.save_model("./model/ispbpe")

    print('FIN')

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()