# Hindi fake news generation using Generative approaches

## GPT approach
1. Train gpt model
```sh
python3 train_gpt.py --dataset=bbc --train_path=bbc_train.csv --test_path=bbc_test.csv --save_path=bloom_heading/  --predict=heading --block_size=512
```
Predict is 'heading' or 'body', depending on what you want the model to predict.

2. Generate fake news using GPT
```sh
python3 fake_news_gpt.py --test_path=bbc_test.csv --ckp=bloom_heading/ --predict=heading --save_path='fake_heading.csv' --max_length=1500
```

## GAN approach
1. Clone the modified github repository
```
git clone https://github.com/div5252/TextGAN-PyTorch
```

2. Get lines from the dataset
```sh
python3 dataset_line.py --input_path=bbc.csv --output_path=bbc_lines.txt --output_path_test=bbc_lines_test.txt
```

3. Run the LeakGAN model on bbc dataset
```sh
cd run
python3 run_leakgan.py 3 0
```

4. Generate fake news using GAN
```sh
python3 fake_news_gan.py --data_path=bbc.csv --samples_path=samples/ --save_path=output/gan_fake.csv --fake_freq=3
```
