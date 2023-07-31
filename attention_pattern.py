import argparse
import torch
import transformers
from functools import partial
from torchvision.io import read_image
from utils import Activation_Patch, get_model, get_value_normed_attention, plot_heatmaps

def main():

    parser = argparse.ArgumentParser(description='attention_pattern.py')
    parser.add_argument('--image', '-i', type=str, default='images/elephant.jpg', help='Path to image')
    parser.add_argument('--model', '-m', type=str, default='ViT-B-16', help='Model name')
    parser.add_argument('--value-normed', '-v', action='store_true', help='Normalize attention patterns using value activations')
    parser.add_argument('--layer', '-l', type=str, default='all', help='Layer to plot')
    parser.add_argument('--head', '-he', type=str, default='all', help='Head to plot')
    parser.add_argument('--save-fig', '-s', action='store_true', help='Save figure')
    parser.add_argument('--save-dir', '-sd', type=str, default='plots', help='Directory to save figure')
    parser.add_argument('--show-fig', '-sh', action='store_true', help='Show figure')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(args.model)

    patch = Activation_Patch()

    for module in model.modules():
        if isinstance(module, transformers.models.vit.modeling_vit.ViTSelfAttention):
            module.register_forward_hook(partial(patch, False, False))

    img = read_image(args.image)
    img = preprocess(img, return_tensors='pt')['pixel_values'].to(device)

    with torch.no_grad():
        logits = model(img)

    attention_heads = patch.activations['attention_heads']
    value_activation_heads = patch.activations['value_activations']
    patch.clear()

    attention_pattern = torch.zeros(12,12,197,197)
    value_activations = torch.zeros(12,12,197,64)
    
    for i in range(12):
        attention_pattern[i] = attention_heads[i]
        value_activations[i] = value_activation_heads[i]

    if args.value_normed:
        value_normed_attention = get_value_normed_attention(attention_pattern, value_activations)
        plot_heatmaps(value_normed_attention, type='value_normed', layer=args.layer, head=args.head, save=args.save_fig, show=args.show_fig, save_dir=args.save_dir)
    else:
        plot_heatmaps(attention_pattern, layer=args.layer, head=args.head, save=args.save_fig, show=args.show_fig, save_dir=args.save_dir)

    

if __name__ == "__main__":
    main()