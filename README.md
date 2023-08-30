# memory-compressed-attention-pytorch
A PyTorch implementation of the Decoder-only Transformer with Memory Compressed Attention for abstractive summarization from "Generating Wikipedia by Summarizing Long Sequences" by Liu et al.

[Paper](https://arxiv.org/abs/1801.10198) </br>
[Dataset](https://arxiv.org/abs/1810.09305)

The main model can be found in `decoder.py`, with it's various layers defined in `decoder.py`, `attention.py` and `utils.py`