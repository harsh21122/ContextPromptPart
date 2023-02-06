# ContextPromptPart

## Dataset 
Dataset for cat only - https://drive.google.com/file/d/1-KPxJS6hh8ofZ3qMOWqNnM__7PacYM3j/view?usp=share_link

## Problem

### Image
Currently, we are stuck at the point where we have a pixel-text score maps using the language-compatible feature map z and the text feature t by:
$$
    s = \hat{z}\hat{t}^{T}, s \in \mathbb{R}^{H_4 \times W_4 \times K}
$$

Above definition is take from DenseCLIP.

Now, that we have s, we can consider it as segmentation in low resolution.
DenseCLIP uses a FPN based decoder to decode images.

We have option to make 2 decoders for the same:
1. [X] FPN decoder
2. [ ] Decoder from SCOPS

### Updates

- [x]**Implemented FPN decoder**
- [x]**Implemented RN50 like CLIP and initialize weights for each layer**
