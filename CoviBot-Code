Source code:
!pip install transformers==4.28.0
!pip install datasets

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from transformers import GPT2LMHeadModel,AutoTokenizer, GPT2TokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset

def process_csv(file_path):
    df = pd.read_csv(file_path)
    qa_pairs = []

    for index, row in df.iterrows():
        question = row['questions']
        answer = row['answers']
        qa_pairs.append(f"Question: {question}\nAnswer: {answer}\n")

    return qa_pairs

def load_dataset(file_path, tokenizer):
    qa_pairs = process_csv(file_path)
    tokenized_dataset = tokenizer(qa_pairs, truncation=True,
                                  padding='max_length', max_length=128,
                                  return_tensors="pt")
    dataset = Dataset.from_dict(tokenized_dataset)
    return dataset

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token


# Load and preprocess the dataset
train_dataset = load_dataset("/content/drive/MyDrive/covid19 Dataset/covid_faq.csv", tokenizer)
valid_dataset = load_dataset("/content/drive/MyDrive/covid19 Dataset/covid_faq.csv", tokenizer)

# Configure and train the model using the Trainer class
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=100,
    save_steps=100,
    warmup_steps=0,
    logging_dir="logs",
    evaluation_strategy="steps",
    save_total_limit=3,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("/content/drive/MyDrive/covid19 Dataset/COVID19_FAQ_model")

# Load the fine-tuned model
fine_tuned_model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/covid19 Dataset/COVID19_FAQ_model")

def ask_question(question, model, tokenizer, max_length=128, num_return_sequences=1):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        early_stopping=True,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()

    # Truncate the answer after the first newline character
    answer = answer.split("\n")[0]

    return answer


# Ask questions using the fine-tuned model
question = "Am I at risk for COVID-19 from mail, packages, or products?"
answer = ask_question(question, fine_tuned_model, tokenizer)
print(f"Question: {question}\nAnswer: {answer}")

question = "Does CDC recommend the use of masks to prevent COVID-19?"
answer = ask_question(question, fine_tuned_model, tokenizer)
print(f"Question: {question}\nAnswer: {answer}")

!pip install torch
!pip install pyTelegramBotAPI

import telebot
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "/content/drive/MyDrive/covid19 Dataset/COVID19_FAQ_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

TOKEN = "5660121550:AAE10YopAM9AcPqV4scWma9jwEl04t7159A"
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    question = message.text.lower()

    if question == "start":
        start_message = "Welcome to the COVID-19 FAQ bot! Please ask a question, and I will provide an answer."
        bot.reply_to(message, start_message)
    else:
        answer = ask_question(question, model, tokenizer)
        bot.reply_to(message, answer)


def ask_question(question, model, tokenizer, max_length=128, num_return_sequences=1):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        early_stopping=True,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()

    # Truncate the answer after the first newline character
    answer = answer.split("\n")[0]

    return answer


bot.polling()

