import torch
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers import TFAutoModel,MBartForConditionalGeneration,AutoModelForSeq2SeqLM
bartpho = AutoModel.from_pretrained("vinai/bartpho-syllable")

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
model=AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable")

print(tokenizer.pad_token_id)
exit()
# print(model.transformer.wte.weight.shape)

line = "Việt Nam là quốc gia có"
gpt = GPT2LMHeadModel.from_pretrained("imthanhlv/gpt2news")
gpt_embedding_size = gpt.transformer.wte
print(type(gpt))
print(gpt_embedding_size)
#get model embedding size
#print(help(model))
bart_embedding_size = model.model.shared.weight.shape[1]

print(bart_embedding_size)

# # tokenizer = GPT2Tokenizer.from_pretrained('NlpHUST/gpt2-vietnamese')
# # model = GPT2LMHeadModel.from_pretrained('NlpHUST/gpt2-vietnamese')

# input_ids = tokenizer.encode(line, return_tensors='pt')
# print(input_ids)
# max_length = 100
# #sample_outputs = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=3)

# sample_outputs = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id,
#                                    do_sample=True,
#                                    max_length=max_length,
#                                    min_length=max_length,
#                                    top_k=40,
#                                    num_beams=5,
#                                    early_stopping=True,
#                                    no_repeat_ngram_size=2,
#                                    num_return_sequences=3)
# for i, sample_output in enumerate(sample_outputs):
#     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))