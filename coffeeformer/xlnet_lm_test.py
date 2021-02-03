from transformers import XLNetTokenizer, XLNetLMHeadModel
import torch
import torch.nn.functional as F
tokenizer = XLNetTokenizer.from_pretrained('./model/spbpe')
model = XLNetLMHeadModel.from_pretrained('./model/spbpe')
model.resize_token_embeddings(len(tokenizer))

tokens = tokenizer.encode("在一件申请需要分案的情<mask>")
# We show how to setup inputs to predict a next token using a bi-directional context.
input_ids = torch.tensor(tokens).unsqueeze(0)  # We will predict the masked token
# print(input_ids)

perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token

target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
# next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
# print(next_token_logits)

predicted_index = torch.argmax(outputs[0][0]).item()
predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
print(predicted_token)



tokens = tokenizer.encode("被侵害人，是因自己的人身、财产、名<mask>")
# We show how to setup inputs to predict a next token using a bi-directional context.
input_ids = torch.tensor(tokens).unsqueeze(0)  # We will predict the masked token
# print(input_ids)

perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token

target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
# next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
# print(next_token_logits)

predicted_index = torch.argmax(outputs[0][0]).item()
predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
print(predicted_token)
