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
1. [ ] FPN decoder
2. [ ] Decoder from SCOPS


### Text

For now, we have decided to follow COoP strategy to improve, segmentation.

One issue that may arise is *" We intend to solve our problem using less/no labelled data. can we make sure that prompts will improve for whatever strategy we use."*

If we decide to do the whole thing in supervised manner? then we can use the CoOP strategy.


If we decide to go with simple template based prompt generation in the beginning.
we will be able to eliminate one learning part from what we currently have.

**Text semantic Constraint**
Making sure that for 2 different images, the text might is same but they should point to the correct parts.

