import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import clip
import glob
import skimage.io as io
import PIL.Image
from tqdm import trange

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
CPU = torch.device("cpu")


def getDevice(deviceID: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    deviceID = min(torch.cuda.device_count() - 1, deviceID)
    return torch.device(f"cuda:{deviceID}")


CUDA = getDevice


class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):
    def getDummyToken(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefixLength, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefixProjections = self.clip_project(
            prefix).view(-1, self.prefixLength, self.gpt_embedding_size)
        embeddingCat = torch.cat((prefixProjections, embedding_text), dim=1)
        if labels is not None:
            dummyToken = self.getDummyToken(tokens.shape[0], tokens.device)
            labels = torch.cat((dummyToken, tokens), dim=1)
        out = self.gpt(inputs_embeds=embeddingCat,
                       labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefixLength: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefixLength = prefixLength
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefixLength > 10:
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefixLength)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size *
                                    prefixLength) // 2, self.gpt_embedding_size * prefixLength))


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entryLength=67,
        top_p=0.8,
        temperature=1.,
        stop_token: str = ".",
):
    model.eval()
    generatedList = []
    stopTokenIndex = tokenizer.encode(stop_token)[0]
    filterValue = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entryIndex in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
            for i in range(entryLength):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / \
                    (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filterValue
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stopTokenIndex == next_token.item():
                    break
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generatedList.append(output_text)
    return generatedList[0]


def ImageCaption():
    model_path = os.path.join("../models", "conceptual_weights.pt")
    device = CUDA(0) if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefixLength = 10
    model = ClipCaptionModel(prefixLength)
    model.load_state_dict(torch.load(model_path, map_location=CPU))
    model = model.eval()
    device = CUDA(0) if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    images = glob.glob("../images/*")
    imagePath = max(images, key=os.path.getctime)
    image = io.imread(imagePath)
    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(
            image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(
            prefix).reshape(1, prefixLength, -1)
    generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    print("\n画像キャプション生成："+generated_text_prefix)
    return generated_text_prefix
