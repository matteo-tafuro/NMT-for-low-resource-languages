import transformers 
import numpy as np
#from transformers import M2M100Tokenizer, M2M100Model
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import wandb

from torch.utils.checkpoint import checkpoint

wandb.login(key="8b579a9be261e9cc153188122605c106eda8322e")


raw_datasets = load_dataset("ted_hrlr", "az_to_en")
metric = load_metric("sacrebleu")

model_checkpoint = "facebook/m2m100_418M"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang="az", tgt_lang="en")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# We have to set the model decoding to force the target language as the bos token. 
model.config.forced_bos_token_id = tokenizer.get_lang_id("en")

if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate English to Romanian: "
else:
    prefix = ""

max_input_length = 128
max_target_length = 128
source_lang = "az"
target_lang = "en"

def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)



# Training setup
batch_size = 4
model_name = "M2M_az_fine-tune_baseline_batch_32.ckpt"
args = Seq2SeqTrainingArguments(
    "checkpoints/"+model_name,
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    optim="adamw_bnb_8bit",
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=500,
    predict_with_generate=True,
    log_level="info",
    logging_dir="logging/M2M_az_fine-tune_baseline_batch_32/",
    logging_strategy="steps",
    logging_steps=46,
    save_strategy="steps",
    save_steps=46, 
    load_best_model_at_end=True, 
    metric_for_best_model="eval_bleu",
    ddp_find_unused_parameters=True,
    fp16=True
)

# Data collator for dynamic padding

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# setup BLEU evaluation metric

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# Initializing Trainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=5)]
)

# start training

trainer.train()

# Evaluate on test set
results_greedy = trainer.predict(tokenized_datasets["test"], num_beams=1)
print("greedy")
print(results_greedy)
results_beam_8 = trainer.predict(tokenized_datasets["test"], num_beams=8)
print("beam = 8)")
print(results_beam_8)

