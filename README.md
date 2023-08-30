# memory-compressed-attention-pytorch
A PyTorch implementation of the Decoder-only Transformer with Memory Compressed Attention for abstractive summarization from "Generating Wikipedia by Summarizing Long Sequences" by Liu et al.

[Paper](https://arxiv.org/abs/1801.10198) </br>
[Dataset](https://arxiv.org/abs/1810.09305)

The main model can be found in `decoder.py`, with it's various layers defined in `decoder.py`, `attention.py` and `utils.py`

# Citations
```
@misc{liu2018generating,
    title={Generating Wikipedia by Summarizing Long Sequences},
    author={Peter J. Liu and Mohammad Saleh and Etienne Pot and Ben Goodrich and Ryan Sepassi and Lukasz Kaiser and Noam Shazeer},
    year={2018},
    eprint={1801.10198},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
@InProceedings{xsum-emnlp,
  author =      "Shashi Narayan and Shay B. Cohen and Mirella Lapata",
  title =       "Don't Give Me the Details, Just the Summary! {T}opic-Aware Convolutional Neural Networks for Extreme Summarization",
  booktitle =   "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing ",
  year =        "2018",
  address =     "Brussels, Belgium",
}
```
