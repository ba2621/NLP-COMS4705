""" This is code for mechanistic interpretability of a gpt-2 style models
	using the TransformerLens library. All the code comes from the following:
	https://www.lesswrong.com/posts/hnzHrdqn3nrjveayv/how-to-transformer-mechanistic-interpretability-in-50-lines
"""


!pip install transformer_lens

# Get a model to play with
from transformer_lens import HookedTransformer, utils
model = HookedTransformer.from_pretrained("gpt2-small")

# Now run the language model
logits = model("What is the capital of Brazil?")

# The logit dimensions are: [batch, position, vocab]
next_token_logits = logits[0, -1]
next_token_prediction = next_token_logits.argmax() 
next_word_prediction = model.tokenizer.decode(next_token_prediction)
print(next_word_prediction) # should be Brasilia


# Now we look at the internal activiations, called "cache"
logits, cache = model.run_with_cache("What is the capital of Brazil?")
for key, value in cache.items():
	print(key, value.shape()) # here we see things like attention pattern, residual streams, and more

# Finding induction heads
# x = "It was so busy. During the holiday season, the mall is always" (test prompt)
# y = " packed" (likely completion)
utils.test_prompt("It was so busy. During the holiday season, the mall is always", " packed", model)
# Now we see the rank of our likely completion, scored by logits and probabilities
# Since we see that its ____, we can try to analyze that behavior.

# Method 1: Residual Stream Patching












