from trl import SFTConfig

def get_training_arguments(output_model, learning_rate):
    return SFTConfig(
        output_dir=output_model,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        num_train_epochs=2,
        max_steps=-1,
        gradient_checkpointing=True,
        warmup_steps=5,
        max_grad_norm=1.0,
        bf16=True,
        push_to_hub=False,     
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        eval_strategy="steps",
        logging_strategy="steps",
        eval_steps=10,
        optim="adamw_torch_fused", 
        packing=False,
        dataset_text_field='text',
        seed=42,
    )
