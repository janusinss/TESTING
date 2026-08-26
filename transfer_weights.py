import torch
import os
import argparse

def map_deblurgan_weights(input_pth, output_pth):
    print(f"Loading DeblurGANv2 weights from {input_pth}...")
    try:
        deblurgan_state = torch.load(input_pth, map_location='cpu')
    except Exception as e:
        print(f"Error loading {input_pth}: {e}")
        return

    # Sometimes weights are stored under a 'model' or 'state_dict' key
    if 'model' in deblurgan_state:
        deblurgan_state = deblurgan_state['model']
    elif 'state_dict' in deblurgan_state:
        deblurgan_state = deblurgan_state['state_dict']

    print(f"Found {len(deblurgan_state.keys())} layers in DeblurGANv2 model.")

    mapped_state = {}
    mapped_count = 0

    for key, weight in deblurgan_state.items():
        new_key = key
        
        # DeblurGANv2 often wraps its FPN MobileNet in a 'generator' or 'model' namespace.
        # We need to strip these so they match our `DGPSynthesizer` namespace.
        prefixes_to_strip = ['module.', 'generator.', 'model.', 'netG.']
        for prefix in prefixes_to_strip:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                
        # Because we use `strict=False` in train.py, any exact matches will be loaded automatically!
        mapped_state[new_key] = weight
        mapped_count += 1

    print(f"Successfully mapped {mapped_count} keys.")
    print(f"Saving compatible weights to {output_pth}...")
    
    torch.save(mapped_state, output_pth)
    print("Done! You can now use this file with train.py --resume_from and --transfer_learning")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map DeblurGANv2 weights to Forensic DGP")
    parser.add_argument("--input", type=str, default="best_fpn.pth", help="Path to downloaded DeblurGANv2 weights")
    parser.add_argument("--output", type=str, default="mapped_deblurgan.pth", help="Output path for compatible weights")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"\n[!] WARNING: {args.input} not found.")
        print("Please download the official fpn_mobilenet.h5 or best_fpn.pth from the DeblurGAN-v2 GitHub and place it here first.")
    else:
        map_deblurgan_weights(args.input, args.output)
