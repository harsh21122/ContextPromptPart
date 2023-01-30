import numpy as np
import cv2
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

device = "cuda" if torch.cuda.is_available() else "cpu"

class PromptLearner(nn.Module):
    def __init__(self, clip_model, partnames):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        # print(partnames)
        partnames.append('background')
        # print(partnames)
        n_cls = len(partnames)
        n_ctx = 15
        ctx_init = ""
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            
            if False:
                # print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                # print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.pr_prefix = " ".join(["X"] * int(n_ctx))
            # self.prompt_prefix2 = " ".join(["X"] * int(n_ctx/3))
            # self.prompt_prefix3 = " ".join(["X"] * int(n_ctx/3))
            # print("ctx_vectors : ", ctx_vectors.shape)
            # print("prompt_prefix : ", self.pr_prefix)

        print(f'Initial context: "{self.pr_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # print("classnames : ", partnames)
        partnames = [name.replace("_", " ") for name in partnames]
        # print("classnames : ", partnames)
        name_lens = [len(_tokenizer.encode(name)) for name in partnames]
        # print("name_lens : ", name_lens)
        prompts = [self.pr_prefix + " " + name + "." for name in partnames]
        # print("prompts : ", prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # print("tokenized_prompts : ", tokenized_prompts[1])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(self.dtype)
        # print("embedding : ", embedding.shape)
# #         print(embedding)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        # print("token_prefix :", embedding[:, :1, :].shape, embedding[:, :1, :])
        # print("token_suffix :", embedding[:, 1 + n_ctx :, :].shape, embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        
        if ctx.dim() == 2:
            # print("ctx : ", ctx.shape)
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            # print("ctx : ", ctx.shape)
        
        # print(classname, partname)
        
        # prompts = []
        # class_name_lens = [len(_tokenizer.encode(name)) for name in classname]
        # part_name_lens = [len(_tokenizer.encode(name)) for name in partname]
        # for idx in range(size):
        #     class_ = classname[idx]
        #     part_ = partname[idx]
        #     # prompt = self.pr_prefix + " " + part_ + " " + self.pr_prefix + " " + class_ + " " + self.pr_prefix + "."
        #     prompt = self.pr_prefix + " " + self.pr_prefix + " " + self.pr_prefix + " " + part_ + " " + class_ + "."
        #     print(prompt)
        #     prompts.append(prompt)
        # print("prompts: ", prompts)

        # self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # print("tokenized_prompts : ", self.tokenized_prompts)
        # with torch.no_grad():
        #     embedding = self.token_embedding(self.tokenized_prompts).type(self.dtype)
        # print("embedding : ", embedding.shape)

        # prefix = embedding[:, :1, :] 
        # suffix = embedding[:, 1 + self.n_ctx :, :]
        # print("prefix : ", prefix.shape)
        # print("suffix : ", suffix.shape)

        # one_third_n_ctx = self.n_ctx // 3
        # print("one_third_n_ctx : ", one_third_n_ctx)

        # new_prompts = []
        # for i in range(size):
        #     class_name_len = class_name_lens[i]
        #     part_name_len = part_name_lens[i]
        #     print("part_name_len class_name_len", part_name_len, class_name_len)

        #     prefix_i = prefix[i : i + 1, :, :]
        #     part_i = suffix[i : i + 1, :part_name_len, :]
        #     class_i = suffix[i : i + 1, part_name_len:class_name_len + part_name_len, :]
        #     suffix_i = suffix[i : i + 1, class_name_len + part_name_len:, :]
        #     ctx_i_1 = ctx[0 : 1, :one_third_n_ctx, :]
        #     ctx_i_2 = ctx[0 : 1, one_third_n_ctx:2*one_third_n_ctx, :]
        #     ctx_i_3 = ctx[0 : 1, 2*one_third_n_ctx:, :]
        #     print(prefix_i.shape, part_i.shape, class_i.shape, suffix_i.shape, ctx_i_1.shape, ctx_i_2.shape, ctx_i_3.shape)
        #     n_prompt = torch.cat(
        #         [
        #             prefix_i,     # (1, 1, dim)
        #             ctx_i_1,  # (1, n_ctx//2, dim)
        #             part_i,      # (1, name_len, dim)
        #             ctx_i_2,  # (1, n_ctx//2, dim)
        #             class_i,
        #             ctx_i_3,
        #             suffix_i,     # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #     print("prompt : ", n_prompt.shape)
        #     new_prompts.append(n_prompt)
        # new_prompts = torch.cat(new_prompts, dim=0)
        # print("prompts : ", new_prompts.shape)

        # return new_prompts

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                # print("prompt : ", prompt.shape)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
            # print("prompts : ", prompts.shape)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        
        # prompts = prompts.unsqueeze(0).expand(size,-1, -1, -1)
        # print("prompts : ", prompts.shape)
        # print(" ctx : ", torch.unique(ctx))
        return prompts

