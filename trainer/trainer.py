# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 15:18:56 2025

@author: gnkhata
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

class CustomTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        data_collator,
        compute_metrics,
        output_dir,
        learning_rate=5e-5,
        weight_decay=0.0,
        epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        generation=False,
        generation_max_length=128,
        generation_num_beams=4,
        device="cuda",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.device = device
        self.generation = generation
        self.generation_max_length = generation_max_length
        self.generation_num_beams = generation_num_beams

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        total_steps = (len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)) * epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=warmup_ratio, total_iters=total_steps
        )

        os.makedirs(output_dir, exist_ok=True)
        self.best_metric = None

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            self.optimizer.zero_grad()

            progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
            for step, batch in progress:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} - Average Train Loss: {avg_loss:.4f}")

            metrics = self.evaluate()
            metric_value = metrics[self.metric_for_best_model]

            if self.best_metric is None or (
                (self.greater_is_better and metric_value > self.best_metric)
                or (not self.greater_is_better and metric_value < self.best_metric)
            ):
                print(f"New best model found ({self.metric_for_best_model}: {metric_value:.4f}). Saving...")
                self.best_metric = metric_value
                self.save_model(epoch)

        print("Training complete!")

    def evaluate(self):
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.generation:
                    generated_tokens = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=self.generation_max_length,
                        num_beams=self.generation_num_beams,
                    )
                    labels = batch["labels"]
                    preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    logits = outputs.logits.detach().cpu().numpy()
                    labels = batch["labels"].detach().cpu().numpy()
                    preds = np.argmax(logits, axis=-1)

                all_preds.extend(preds)
                all_labels.extend(labels)

        metrics = self.compute_metrics((all_preds, all_labels))
        metrics["eval_loss"] = total_loss / len(eval_loader)
        print(f"Evaluation metrics: {metrics}")
        return metrics

    def save_model(self, epoch):
        save_dir = os.path.join(self.output_dir, f"best_model_epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
