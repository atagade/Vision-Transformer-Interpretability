import torch
import torch.nn as nn
import math
import einops
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor

class Activation_Patch:
  def __init__(self):
    self.activations = {}
    self.activations['attention_heads'] = []
    self.activations['value_activations'] = []

  def __call__(self, patch_head, patch_activation, module, module_in, module_out):
    self.activations['input'] = (module_in)
    self.activations['output'] = (module_out)
    self.activations['key'] = module.key
    self.activations['query'] = module.query

    def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
      new_x_shape = x.size()[:-1] + (12, 64)
      x = x.view(new_x_shape)
      return x.permute(0, 2, 1, 3)

    key_activations = transpose_for_scores(module.key(module_in[0]))
    query_activations = transpose_for_scores(module.query(module_in[0]))
    value_activations = transpose_for_scores(module.value(module_in[0]))

    self.activations['value_activations'].append(value_activations)

    attention_scores = torch.matmul(query_activations, key_activations.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(64)
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    if isinstance(patch_activation, torch.Tensor):
      attention_probs[:,patch_head,:,:] = patch_activation
      self.activations['patching_done'] = True
    else:
      self.activations['patching_done'] = False

    self.activations['attention_heads'].append(attention_probs)

    # Mask heads if we want to
    if module_in[1] is not None:
        attention_probs = attention_probs * module_in[1]

    context_layer = torch.matmul(attention_probs, value_activations)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (12*64,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if module_in[2] else (context_layer,)

    return outputs

  def clear(self):
    self.activations = {}

def get_model(model: str):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model == 'ViT-B-16' or model.lower() == 'vb16':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device).eval()
        preprocess = ViTImageProcessor()
    else:
        raise ValueError(f"Model {model} not found")
    
    return model, preprocess

def get_value_normed_attention(attention_pattern, value_activations):
   
    value_normed = einops.repeat(value_activations.norm(dim=3), 'a b c -> a b repeat c', repeat=197)
    max_values_per_head = einops.repeat(value_normed.max(dim=3).values, 'a b c -> a b repeat c', repeat=197)
    value_normalised = value_normed/max_values_per_head
    normed_attention = value_normalised * attention_pattern
    return normed_attention

def plot_heatmaps(attention_pattern, show=False, save=True, save_dir='plots', layer='all', head='all', type='attention_pattern'):
   
    patches = [f'P{i}' for i in range(197)]
    layers = [f'L{i}' for i in range(12)]
    heads = [f'H{i}' for i in range(12)]

    if layer == 'all':
        layer = [i for i in range(12)]
    else:
       layer = [int(layer)]
    
    if head == 'all':
        head = [i for i in range(12)]
    else:
        head = [int(head)]

    for i in range(len(layer)):
            
        for j in range(len(head)):
        
            fig, ax = plt.subplots(figsize=(50,50))
            im = ax.imshow(attention_pattern[layer[i], head[j], :, :], cmap = 'Greys')

            ax.set_yticks(np.arange(len(patches)), labels=patches)
            ax.set_xticks(np.arange(len(patches)), labels=patches)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Attention', rotation=-90, va="bottom")


            fig.tight_layout()

            if save:
                if type == 'attention_pattern':
                    ax.set_title(f'L{layer[i]}_H{head[j]}: Attention Pattern')
                    plt.savefig(f'{save_dir}/L{layer[i]}_H{head[j]}_attention_pattern.png')
                elif type == 'value_normed':
                    ax.set_title(f'L{layer[i]}_H{head[j]}: Value-normed Attention')
                    plt.savefig(f'{save_dir}/L{layer[i]}_H{head[j]}_value_normed_attention.png')
            elif show:
                plt.show()

            plt.close(fig)
