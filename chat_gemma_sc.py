import gradio as gr
from transformers import AutoTokenizer,pipeline, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model_id = "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipeline = pipeline("text-generation", model=model_id, 
                    model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": quantization_config,}, )


def chat_with_model(msg, history):
    messages = []
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": msg})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(prompt, max_new_tokens=256)[0]["generated_text"][len(prompt):]
    return outputs
    

gr.ChatInterface(fn=chat_with_model, examples=[
    ["Hello there! How are you doing?"],
    ["Can you explain briefly to me what is the Python programming language?"],
    ["Explain the plot of Cinderella in a sentence."],
    ["How many hours does it take to train an AI model like you?"],
    ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ["Hello, solve for x: 4x + 25 = 0"],
    ["Describe opentelemetry and how it works."],
    ["Describe how the stock market works."],
    ["Give me good ideas on what books to read to become a successful investor"],
    ["Write a python code to start a webserver and serve a page saying Hello World!, instrument the code with opentelemetry."],
], title="Chat with Gemma").launch()
