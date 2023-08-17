# Vision-Transformer-Interpretability

## Attention Patterns
Currently only support ViT-B-16 from Huggingface

Attention patterns for all layers and heads:
`python attention_pattern.py -m vb16 -i PATH_TO_IMAGE`

Attention patterns for a particular layer and head:
`python attention_pattern.py -m vb16 -i PATH_TO_IMAGE -l LAYER -he HEAD`
