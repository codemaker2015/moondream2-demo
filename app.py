from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)


# 1. Image Captioning
image = Image.open('./inputs/street.jpg')
enc_image = model.encode_image(image)
input = "Describe the image."
output = model.answer_question(enc_image, input, tokenizer)
print(f"\nInput: {input}\nOutput: {output}")


# 2. Visual Question-Answering
image = Image.open('./inputs/cats.jpg')
enc_image = model.encode_image(image)
input = "How many cats the girl is holding?"
output = model.answer_question(enc_image, input, tokenizer)
print(f"\nInput: {input}\nOutput: {output}")
input = "What is their color?"
output = model.answer_question(enc_image, input, tokenizer)
print(f"\nInput: {input}\nOutput: {output}")


# 3. Visual Knowledge Reasoning
image = Image.open('./inputs/tajmahal.jpg')
enc_image = model.encode_image(image)
input = "Tell about the history of this place"
output = model.answer_question(enc_image, input, tokenizer)
print(f"\nInput: {input}\nOutput: {output}")


# 4. Visual Contextual Understanding
image = Image.open('./inputs/boy_playing_with_pet.jpg')
enc_image = model.encode_image(image)
input = "How does the boy feel and why?"
output = model.answer_question(enc_image, input, tokenizer)
print(f"\nInput: {input}\nOutput: {output}")

# # 5. Text Recognition
image = Image.open('./inputs/written_quote.jpg')
enc_image = model.encode_image(image)
input = "What's written on this piece of paper?"
output = model.answer_question(enc_image, input, tokenizer)
print(f"\nInput: {input}\nOutput: {output}")