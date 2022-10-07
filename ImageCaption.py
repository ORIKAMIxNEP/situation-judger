import glob
import os
from typing import List, Optional, Tuple, Union

import clip
import numpy as np
import PIL.Image
import skimage.io as io
import torch
import torch.nn.functional as nnf
from torch import nn
from tqdm import trange
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
    def getDummyToken(self, batchSize: int, device: D) -> T:
        return torch.zeros(batchSize, self.prefixLength, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embeddingText = self.gpt.transformer.wte(tokens)
        prefixProjections = self.clip_project(
            prefix).view(-1, self.prefixLength, self.gpt_embedding_size)
        embeddingCat = torch.cat((prefixProjections, embeddingText), dim=1)
        if labels is not None:
            dummyToken = self.getDummyToken(tokens.shape[0], tokens.device)
            labels = torch.cat((dummyToken, tokens), dim=1)
        out = self.gpt(inputs_embeds=embeddingCat,
                       labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefixLength: int, prefixSize: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefixLength = prefixLength
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefixLength > 10:
            self.clip_project = nn.Linear(
                prefixSize, self.gpt_embedding_size * prefixLength)
        else:
            self.clip_project = MLP(
                (prefixSize, (self.gpt_embedding_size * prefixLength) // 2, self.gpt_embedding_size * prefixLength))


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
        entryCount=1,
        entryLength=67,
        topP=0.8,
        temperature=1.,
        stopToken: str = ".",
):
    model.eval()
    generatedList = []
    stopTokenIndex = tokenizer.encode(stopToken)[0]
    filterValue = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entryIndex in trange(entryCount):
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
                sortedLogits, sortedIndices = torch.sort(
                    logits, descending=True)
                cumulativeProbs = torch.cumsum(
                    nnf.softmax(sortedLogits, dim=-1), dim=-1)
                sortedIndicesToRemove = cumulativeProbs > topP
                sortedIndicesToRemove[...,
                                      1:] = sortedIndicesToRemove[..., :-1].clone()
                sortedIndicesToRemove[..., 0] = 0
                indicesToRemove = sortedIndices[sortedIndicesToRemove]
                logits[:, indicesToRemove] = filterValue
                nextToken = torch.argmax(logits, -1).unsqueeze(0)
                nextTokenEmbed = model.gpt.transformer.wte(nextToken)
                if tokens is None:
                    tokens = nextToken
                else:
                    tokens = torch.cat((tokens, nextToken), dim=1)
                generated = torch.cat((generated, nextTokenEmbed), dim=1)
                if stopTokenIndex == nextToken.item():
                    break
            outputList = list(tokens.squeeze().cpu().numpy())
            outputText = tokenizer.decode(outputList)
            generatedList.append(outputText)
    return generatedList[0]


def ImageCaption():
    # 画像からキャプションを生成する関数
    modelPath = os.path.join("../models", "conceptual_weights.pt")
    device = CUDA(0) if torch.cuda.is_available() else "cpu"
    clipModel, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefixLength = 10
    model = ClipCaptionModel(prefixLength)
    model.load_state_dict(torch.load(modelPath, map_location=CPU))
    model = model.eval()
    device = CUDA(0) if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    images = glob.glob("../images/*")
    imagePath = max(images, key=os.path.getctime)
    image = io.imread(imagePath)
    pilImage = PIL.Image.fromarray(image)
    image = preprocess(pilImage).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clipModel.encode_image(image).to(device, dtype=torch.float32)
        prefixEmbed = model.clip_project(prefix).reshape(1, prefixLength, -1)

    generatedTextPrefix = generate2(model, tokenizer, embed=prefixEmbed)
    print("\n画像のキャプション："+generatedTextPrefix)
    return generatedTextPrefix
