import argparse
import os
import pandas as pd
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--samples_path', type=str, required=True, help='Path for the GAN generated samples')
parser.add_argument('--data_path', type=str, required=True, help='Path of input dataset')
parser.add_argument('--fake_freq', type=int, default=5, help='Lower number represents more fakeness')
parser.add_argument('--save_path', type=str, required=True, help='Output csv path')
opt = parser.parse_args()

def clean_line(line):
    cleaned_line = line
    if cleaned_line[-1] == '\n':
        cleaned_line = cleaned_line[:-1]
    punctuation = ['\"', '\'', '(', ')', ':']
    cleaned_line = "".join([i for i in cleaned_line if i not in punctuation])
    return cleaned_line


def fake_body(body, gan_lines):
    lines = body.split('.')
    new_body = ""
    for line in lines:
        rand = random.randint(1, opt.fake_freq)
        if rand == 1:
            rand_index = random.randint(0, len(gan_lines) - 1)
            new_body += gan_lines[rand_index] + '.'
        else:
            new_body += line + '.'

    return new_body


if __name__ == '__main__':
    random.seed(0)

    gan_lines = []
    for fname in os.listdir(opt.samples_path):
        with open(os.path.join(opt.samples_path, fname), 'r') as file:
            for line in file:
                cleaned_line = clean_line(line)
                if len(cleaned_line) >= 3 and cleaned_line[0] != '\n':
                    gan_lines.append(cleaned_line)

    df = pd.read_csv(opt.data_path)
    new_df = pd.DataFrame(columns=['body', 'heading', 'label'])
    itr = 0
    for i in tqdm(range(len(df))):
        new_df.loc[itr] = df.loc[i]
        itr += 1
        gan_body = fake_body(str(df['body'][i]), gan_lines)
        new_df.loc[itr] = [gan_body, df['heading'][i], 1]
        itr += 1
    
    new_df.dropna(inplace=True)
    new_df.to_csv(opt.save_path)