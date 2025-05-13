import torch
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    TrainerCallback, 
    TrainerControl, 
    TrainerState
)
import json
import mlflow
import os
from transformers.integrations import HfDeepSpeedConfig
import shutil
import torch.distributed as dist

def process_func(example):

    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 448,
                    "resized_width": 448,
                },
                {"type": "text", "text": "Is this image manipulated or synthesized?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) 
    image_inputs, video_inputs = process_vision_info(messages)  
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

class SaveBestTrainLossCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.best_loss = float("inf")
        self.output_dir = output_dir

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            current_loss = logs["loss"]
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                kwargs["model"].save_pretrained(self.output_dir)
                tokenizer.save_pretrained(self.output_dir)
                processor.save_pretrained(self.output_dir)
                print(f"Saved new best model with train loss: {current_loss:.4f}")

if __name__ == "__main__":

    mlflow.set_experiment("Qwen2.5-VL-3B-Instruct-Fintune-Single-GPU-LoRA-Sample")

    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 4,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,  
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }
    
    # Initialize DeepSpeed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"


    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id)


    dschf = HfDeepSpeedConfig(ds_config)
    

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    model.enable_input_require_grads() 

    train_ds = Dataset.from_json("./train.json")
    train_ds = train_ds.select(range(100))
    train_dataset = train_ds.map(process_func)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  
        r=64, 
        lora_alpha=16,  
        lora_dropout=0.05,  
        bias="none",
    )

    peft_model = get_peft_model(model, config)

    args = TrainingArguments(
        output_dir="./output/Qwen2.5-VL-3B-Instruct",
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=4,  
        logging_steps=10,
        logging_first_step=5,
        num_train_epochs=1,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="mlflow",
        bf16=True,
        deepspeed=ds_config, 
    )
    
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[SaveBestTrainLossCallback(output_dir="./output/Qwen2.5-VL-3B-Instruct/best_step")]
    )

    
    with mlflow.start_run(log_system_metrics=True) as run:
        trainer.train()
        save_path = './output/Qwen2.5-VL-3B-Instruct/best_step'
        directory = os.path.dirname(save_path)
        shutil.make_archive(os.path.join(directory, "best_step"), 'zip', save_path)
        mlflow.log_artifact(os.path.join(directory, "best_step.zip"))
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/best_step.zip",
            name="Qwen2.5-VL-3B-Instruct-Fintune-Single-GPU-LoRA-Sample"
        )