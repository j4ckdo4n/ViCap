import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup,MBartForConditionalGeneration
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from IPython.display import Image
from transformers import AutoTokenizer
from PIL import ImageFile, Image
from transformers.models.mbart.modeling_mbart import shift_tokens_right
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from glob import glob
import sys

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device(0)

# current_directory = os.getcwd()
# save_path = os.path.join(os.path.dirname(current_directory), "pretrained_models")
# os.makedirs(save_path, exist_ok=True)
# model_path = os.path.join(save_path, 'model_wieghts.pt')

class MLP(nn.Module):
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.model(x)
    def __init__(self,sizes:Tuple[int,...],bias=True,act=nn.Tanh):
        super().__init__()
        """Project clip output to embedding of first prefix_length tokens"""
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                # added some dropout here
                layers.append(nn.Dropout(p=0.2))
        self.model = nn.Sequential(*layers)
class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate prefix tokens, shape Bxprefix_length"""
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        embedding_text = self.bart.get_input_embeddings()(tokens).squeeze()
        decoder_input_ids = shift_tokens_right(tokens, 1)
        decoder_inputs_embeds = self.bart.get_input_embeddings()(decoder_input_ids).squeeze()
        #embedding_text = self.bart.model.shared(tokens)
       # print(type(embedding_text))
        #print(embedding_text)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.bart_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        # print("embedding_cat",embedding_cat.shape)
        # #print(embedding_cat.shape)
        # print("clone")
        # #print(embedding_cat.clone())
        # print(type(embedding_cat))
        # print(mask.shape)
        #device=torch.device("cuda:0")

        out = self.bart(inputs_embeds=embedding_cat ,decoder_inputs_embeds=decoder_inputs_embeds, labels=labels, attention_mask=mask)

        return out

    def __init__(self, prefix_length: int = 10, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.bart = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable")
        self.bart_embedding_size = self.bart.model.shared.weight.shape[1]

        self.clip_project = MLP(
            (
                prefix_size,
                (self.bart_embedding_size * prefix_length) // 2,
                self.bart_embedding_size * prefix_length,
            )
        )
class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.bart.eval()
        return self




def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
#     stop_token_index = tokenizer.encode(stop_token)[0]
    stop_token_index = 0

    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

#@title GPU/CPU

is_gpu = True #@param {type:"boolean"}

#@title CLIP model + GPT2 tokenizer

device = CUDA
clip_model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, prefix=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
#     stop_token_index = 0

    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    embed=model.clip_project(prefix).reshape(1, prefix_length, -1)
    with torch.no_grad():
        #print(embed)
        print(tokens)
        print(prompt)
        if embed is not None:
            generated = embed
            #tokens = torch.tensor(tokenizer.encode("Một người đang cầm một lọ nhựa và một viên thuốc trên tay."))
            #tokens = tokens.unsqueeze(0).to(device)
            decoder_input_ids = shift_tokens_right(prefix, 1)
            #conver decoder_input_ids to long tensor
            decoder_input_ids = torch.tensor(decoder_input_ids).long()
            decoder_inputs_embeds = model.bart.get_input_embeddings()(decoder_input_ids).squeeze()
        else:

            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                embedding_text = model.bart.get_input_embeddings()(tokens).squeeze()
                decoder_input_ids = shift_tokens_right(tokens, 1)
                decoder_inputs_embeds = model.bart.get_input_embeddings()(decoder_input_ids).squeeze()
                #generated = model.bart(inputs_embeds=embedding_text,decoder_inputs_embeds=decoder_inputs_embeds)
                print("hello")
                print(decoder_inputs_embeds.shape)
                generated=embedding_text
        for i in range(entry_length):
            print(generated.shape)
            print(decoder_inputs_embeds.shape)
            outputs=model.bart(inputs_embeds=generated,decoder_inputs_embeds=decoder_inputs_embeds)
            logits = outputs.logits
            print("logits")
            print(logits.shape)
            print("is_stopped")
            print(is_stopped.shape)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                print(logits.shape)
                print(is_stopped)
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.bart.get_input_embeddings()(next_tokens).squeeze().view(generated.shape[0],1,-1)
            # next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            # decoder_input_ids = shift_tokens_right(tokens, 1)
            # decoder_inputs_embeds = model.bart.get_input_embeddings()(decoder_input_ids).squeeze()
            print("decoder_inputs_embeds",decoder_inputs_embeds.shape)
            print("next_token")
            print(is_stopped.shape)

            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

prefix_length = 10

# model = ClipCaptionModel(prefix_length)
model = ClipCaptionPrefix(prefix_length)
model_path = sys.argv[1]


model.load_state_dict(torch.load(model_path, map_location=CPU))

model = model.eval()
device = CUDA
# model_path = "./nmt-015-x3b16.pt"
model = model.to(device)

out = []

embeddings = []
ids = []


test = False
for filename in tqdm(sorted(glob("viecap4h-public-train/images/*"))):
# test = True
# for filename in sorted(glob("./viecap/images_public_test/*"))[:20]:
    # image = io.imread(UPLOADED_FILE)
    # # pil_image = PIL.Image.fromarray(image)
    # # pil_img = Image(filename=UPLOADED_FILE)
    pil_img = Image.open(filename)
    print(filename)
    image = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        # if type(model) is ClipCaptionE2E:
        #     prefix_embed = model.forward_image(image)
        # else:
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        generated_text_prefix = generate_beam(model, tokenizer, prefix=prefix, temperature=1, beam_size=10)[0]

    # if test:
    #     display(pil_img)
    #     print(generated_text_prefix)
    # if use_beam_search:
    # else:
    #     generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    out.append({
        "id": os.path.split(filename)[-1],
        "captions": generated_text_prefix
    })

    embeddings.append(prefix)
    ids.append(os.path.split(filename)[-1])

import json

# !rm *.json
# !rm *.zip

sub = {i["id"]: i["captions"] for i in out}
with open("results.json", "w") as f:
    json.dump(sub, f, ensure_ascii=False)
# sample = json.load(open("vietcap4h-public-test/sample_submission.json", "r"))
# for i in sample:
#     i["captions"] = sub[i["id"]].split("\n")[0]


# with open("results.json", "w") as f:
#     json.dump(sample, f, ensure_ascii=False)

# !zip -9r submission.zip results.json
# !curl --upload-file ./submission.zip https://transfer.sh/submission.zip