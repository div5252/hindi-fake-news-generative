from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, required=True, help='Path of input test dataset')
parser.add_argument('--max_length', type=int, default=2000, help='Max length of the generated news')
parser.add_argument('--ckp', type=str, required=True, help='Checkpoint for finetuned model')
parser.add_argument('--predict', type=str, default='body', help='Predict: [heading, body]')
parser.add_argument('--save_freq', type=int, default=50, help='Frequency for saving generated articles')
parser.add_argument('--save_path', type=str, required=True, help='Output csv path')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
opt = parser.parse_args()

model_checkpoint = opt.ckp
tokenizer_checkpoint = "bigscience/bloom-1b1"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

delimiter = '#'

if __name__ == '__main__':
    test = pd.read_csv(opt.test_path)

    text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device='cuda:0')

    new_df = pd.DataFrame(columns=['body', 'heading', 'label'])
    itr = 0
    prompts = []
    bodies = []
    headings = []
    for i in tqdm(range(len(test))):
        if opt.predict == 'heading':
            prompt = 'Body: ' + str(test['body'][i]) + '. Headline: '
        else:
            prompt = 'Headline: ' + str(test['heading'][i]) + '. Body: '
        prompts.append(prompt)
        bodies.append(str(test['body'][i]))
        headings.append(str(test['heading'][i]))

        if i % opt.save_freq == (opt.save_freq - 1):
            print("Writing batch " + str(i // opt.save_freq))
            generated_texts = text_generation(prompts, max_length=opt.max_length, do_sample=True)

            for j, generated_text in enumerate(generated_texts):
                new_df.loc[itr] = [bodies[j], headings[j], 0]
                itr += 1
                output = generated_text[0]['generated_text']

                if opt.predict == 'heading':
                    start = output.find('Headline: ')
                    output = output[start + len('Headline:'):]
                    end = output.find('Body:')
                    output = output[:end].strip()
                    new_df.loc[itr] = [bodies[j], output, 1]
                else:
                    start = output.find('Body: ')
                    output = output[start + len('Body:'):]
                    end = output.find('Headline:')
                    output = output[:end].strip()
                    new_df.loc[itr] = [output, headings[j], 1]
                itr += 1

            new_df.to_csv(opt.save_path)
            prompts = []
            bodies = []
            headings = []


    # Last batch
    generated_texts = text_generation(prompts, max_length=opt.max_length, do_sample=True)

    for j, generated_text in enumerate(generated_texts):
        new_df.loc[itr] = [bodies[j], headings[j], 0]
        itr += 1
        output = generated_text[0]['generated_text']

        if opt.predict == 'heading':
            start = output.find('Headline: ')
            output = output[start + len('Headline:'):]
            end = output.find('Body:')
            output = output[:end].strip()
            new_df.loc[itr] = [bodies[j], output, 1]
        else:
            start = output.find('Body: ')
            output = output[start + len('Body:'):]
            end = output.find('Headline:')
            output = output[:end].strip()
            new_df.loc[itr] = [output, headings[j], 1]
        itr += 1

    new_df.to_csv(opt.save_path)