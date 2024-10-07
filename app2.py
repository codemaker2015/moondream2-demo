from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)


def process_data(path, prompt, model=model,tokenizer=tokenizer):
    image = Image.open(path)
    enc_image = model.encode_image(image)
    return model.answer_question(enc_image, prompt, tokenizer)

# 1. Image Captioning
input = "Describe the image."
output = process_data('./inputs/street.jpg', input)
print(f"\nInput: {input}\nOutput: {output}")

# 2. Visual Question-Answering
input = "How many cats the girl is holding?"
output = process_data('./inputs/cats.jpg', input)
print(f"\nInput: {input}\nOutput: {output}")
input = "What is their color?"
output = process_data('./inputs/cats.jpg', input)
print(f"\nInput: {input}\nOutput: {output}")

# 3. Visual Knowledge Reasoning
input = "Tell about the history of this place"
output = process_data('./inputs/tajmahal.jpg', input)
print(f"\nInput: {input}\nOutput: {output}")

# 4. Visual Contextual Understanding
input = "How does the boy feel and why?"
output = process_data('./inputs/boy_playing_with_pet.jpg', input)
print(f"\nInput: {input}\nOutput: {output}")

# 5. Text Recognition
input = "What's written on this piece of paper?"
output = process_data('./inputs/written_quote.jpg', input)
print(f"\nInput: {input}\nOutput: {output}")