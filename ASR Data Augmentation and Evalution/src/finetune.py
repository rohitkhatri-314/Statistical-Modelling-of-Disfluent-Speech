import argparse
import torch
import soundfile as sf
from datasets import Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Union
from utils import read_jsonl

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def load_data(manifest_path, processor):
    records = list(read_jsonl(manifest_path))
    
    def prepare_dataset(batch):
        audio_path = batch.get("stuttered_audio_path")
        if not audio_path:
            # Fallback for synthetic generated dataset layout
            audio_path = batch.get("clean_audio_path")
            
        text = batch.get("stuttered_text", batch.get("clean_text", ""))
        
        speech, sample_rate = sf.read(audio_path)
        # Wav2Vec2 expects 16kHz
        batch["input_values"] = processor(speech, sampling_rate=16000).input_values[0]
        batch["labels"] = processor.tokenizer(text).input_ids
        return batch

    dataset = Dataset.from_list(records)
    # Using num_proc=1 for safe cross-platform compatibility
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1)
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 on augmented stuttered speech.")
    parser.add_argument("--manifest", required=True, help="Path to augmented training manifest (jsonl).")
    parser.add_argument("--eval_manifest", default=None, help="Path to evaluation manifest (optional).")
    parser.add_argument("--model_name", default="facebook/wav2vec2-base-960h", help="Base model.")
    parser.add_argument("--output_dir", default="./outputs/wav2vec2-stuttered", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    
    args = parser.parse_args()

    print(f"Loading processor and model from: {args.model_name}")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    
    # Freeze CNN feature extractor to save memory and compute
    model.freeze_feature_extractor()

    print(f"Loading training data from: {args.manifest}")
    train_dataset = load_data(args.manifest, processor)
    eval_dataset = load_data(args.eval_manifest, processor) if args.eval_manifest else None

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        eval_strategy="steps" if eval_dataset else "no",
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        eval_steps=500 if eval_dataset else None,
        logging_steps=100,
        learning_rate=args.learning_rate,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving fine-tuned model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
