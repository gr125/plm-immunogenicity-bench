from transformers import BertModel, BertTokenizer
import re

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")
sequence_Example = "A E T C Z A O"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt') # CLS = 2 SEP = 3
print(encoded_input)
output = model(**encoded_input)
print(output.last_hidden_state)
print(output.last_hidden_state.shape)
