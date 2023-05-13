from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import argparse
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset: [bbc, navbharat]')
parser.add_argument('--train_path', type=str, required=True, help='Path of input train dataset')
parser.add_argument('--test_path', type=str, required=True, help='Path of input test dataset')
parser.add_argument('--block_size', type=int, default=256, help='Block size for finetuning')
parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--save_path', type=str, required=True, help='Path for saving finetuned model')
parser.add_argument('--predict', type=str, default='body', help='Predict: [heading, body]')
opt = parser.parse_args()

delimiter = '#'

def combine_head_body(df):
    def combine(row):
        return 'Headline: ' + str(row['heading']) + '. Body: ' + str(row['body']) + delimiter

    df['text'] = df.apply(combine, axis=1)


def combine_body_head(df):
    def combine(row):
        return 'Body: ' + str(row['body']) + '. Headline: ' + str(row['heading']) + delimiter

    df['text'] = df.apply(combine, axis=1)


def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # concatenated_examples = examples
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // opt.block_size) * opt.block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + opt.block_size] for i in range(0, total_length, opt.block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == '__main__':
    if opt.predict == 'heading':
        combine_fn = combine_body_head
    else:
        combine_fn = combine_head_body

    train = pd.read_csv(opt.train_path)
    test = pd.read_csv(opt.test_path)
    combine_fn(train)
    combine_fn(test)

    dataset_train = Dataset.from_pandas(train)
    dataset_test = Dataset.from_pandas(test)

    model_checkpoint = "bigscience/bloom-1b1"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_dataset_train = dataset_train.map(tokenize_function, batched=True, num_proc=4, remove_columns=['Unnamed: 0','body', 'heading', 'label', 'text'])
    tokenized_dataset_test = dataset_test.map(tokenize_function, batched=True, num_proc=4, remove_columns=['Unnamed: 0','body', 'heading', 'label', 'text'])
    lm_dataset_train = tokenized_dataset_train.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    lm_dataset_test = tokenized_dataset_test.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        f"bloom-finetuned-" + str(opt.predict),
        evaluation_strategy = "epoch",
        learning_rate=opt.lr,
        weight_decay=opt.weight_decay,
        num_train_epochs=opt.n_epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset_train,
        eval_dataset=lm_dataset_test,
    )

    trainer.train()
    trainer.save_model(opt.save_path)
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")