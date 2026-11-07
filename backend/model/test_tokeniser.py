
from tokenizer import char_tokenizer

text = "hello world"

# Create tokenizer
tokenizer = char_tokenizer(text)

# Display vocabulary
print("Vocabulary:", tokenizer.chars)
print("Vocabulary Size:", tokenizer.vocab_size)

# Encode
encoded = tokenizer.encode("hello")
print("Encoded:", encoded)

# Decode
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
