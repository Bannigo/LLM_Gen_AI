import os
import torch
from transformers import MarianMTModel, MarianTokenizer

def download_model(model_name="Helsinki-NLP/opus-mt-de-en", model_dir="./model_directory"):
    """
    Checks if the model exists in `model_dir`. If not, downloads it from Hugging Face.
    """
    # Ensure the model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Check if the model and tokenizer are already downloaded
    model_path = os.path.join(model_dir, model_name.replace("/", "_"))
    if os.path.isdir(model_path):
        print(f"Model already exists in {model_path}. Loading from cache.")
    else:
        print(f"Downloading model to {model_path}")
    
    # Load tokenizer and model with custom cache directory
    tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=model_path)
    model = MarianMTModel.from_pretrained(model_name, cache_dir=model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer

def translate_text(model, tokenizer, text, source_lang="German", target_lang="English"):
    """
    Translates text from source language to target language using the specified model.
    """
    # Optional prompt for context
    prompt = f"Translating from {source_lang} to {target_lang}: {text}"
    print(prompt)
    
    # Tokenize and encode the input text
    encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Move the input tensors to the same device as the model
    device = model.device
    encoded_text = {key: value.to(device) for key, value in encoded_text.items()}
    
    # Generate the translation
    with torch.no_grad():  # Disable gradients for faster inference
        translated_tokens = model.generate(
            **encoded_text,
            max_length=300,  # Set a reasonable max length for the output
            num_beams=1,     # Greedy decoding
            do_sample=False  # Deterministic output
        )
    
    # Decode the translated tokens to get the final translated text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def main():
    # Step 1: Load the model and tokenizer
    model_dir = "./model_directory"  # Specify where the model should be saved
    model, tokenizer = download_model(model_dir=model_dir)

    # Step 2: Accept prompt input from the user
    source_text = input("Enter the text to translate: ")
    
    # Step 3: Translate the text
    translated_text = translate_text(model, tokenizer, source_text)
    
    # Step 4: Display the translation
    print("Translated text:", translated_text)

if __name__ == "__main__":
    main()
