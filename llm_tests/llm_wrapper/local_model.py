from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch

class LocalModel:
    def __init__(self, model_id="gpt2", lora=False):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager")
        self.lora = lora
        if lora:
            self.model = prepare_model_for_kbit_training(self.model)
            config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.1)
            self.model = get_peft_model(self.model, config)

    def tokenize(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    def complete(self, prompt, max_new_tokens=50):
        inputs = self.tokenize(prompt)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_attention_and_logits(self, prompt):
        inputs = self.tokenize(prompt)
        outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        return {
            "logits": outputs.logits,
            "attentions": outputs.attentions,
            "tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        }

    def analyze_last(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        tokens = [f"{t:<6}".replace("Ġ", "\u2423") for t in self.tokenizer.convert_ids_to_tokens(input_ids[0])]

        print("Prompt tokens:")
        print("         " + "".join(tokens))

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        last_attn = outputs.attentions[-1]  # [batch, heads, tgt, src]
        final_token_idx = input_ids.shape[-1] - 1

        print("\nAttention for last input token:")
        for head_id, head_attn in enumerate(last_attn[0]):
            attn_row = head_attn[final_token_idx]  # [src_len]
            weights = attn_row.tolist()
            row = f"Head {head_id:2d}:"
            row += " ".join(f"{w:5.2f}" for w in weights)
            print(row)

    def analyze_next(self, prompt, topk=3):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        tokens = [f"{t:<6}".replace("Ġ", "\u2423") for t in self.tokenizer.convert_ids_to_tokens(input_ids[0])]

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        topk_ids = torch.topk(probs, topk).indices.tolist()
        topk_tokens = [t.replace("Ġ", "\u2423") for t in self.tokenizer.convert_ids_to_tokens(topk_ids)]

        print("\nNext token prediction:")
        print("         " + "".join(tokens + ["(" + " ".join(topk_tokens) + ")"]))

        extended_ids = torch.cat([input_ids, torch.tensor([[topk_ids[0]]])], dim=1)
        with torch.no_grad():
            ext_outputs = self.model(input_ids=extended_ids, output_attentions=True)

        ext_attn = ext_outputs.attentions[-1]
        print("\nAttention used for prediction:")
        for head_id, head_attn in enumerate(ext_attn[0]):
            attn_row = head_attn[-1]  # [src_len]
            weights = attn_row.tolist()
            row = f"Head {head_id:2d}:"
            row += " ".join(f"{w:5.2f}" for w in weights)
            print(row)

    def fine_tune(self, dataset, output_dir="./finetuned_model"):
        args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=50,
            save_total_limit=1,
            use_cpu=True
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train()
