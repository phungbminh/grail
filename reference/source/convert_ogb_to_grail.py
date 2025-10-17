import torch
import os
from tqdm import tqdm

def convert_split(split_name, input_dir, output_dir):
    """
    Loads a .pt file for a given split, extracts the triplets,
    and writes them to a .txt file in the format GraIL expects.
    """
    input_file = os.path.join(input_dir, f'{split_name}.pt')
    output_file = os.path.join(output_dir, f'{split_name}.txt')

    print(f"Processing {input_file} -> {output_file}...")

    try:
        # Load the .pt file, allowing it to unpickle NumPy objects
        data = torch.load(input_file, weights_only=False)

        # Extract the relevant tensors
        heads = data['head']
        relations = data['relation']
        tails = data['tail']

        # Write to the output .txt file
        with open(output_file, 'w') as f:
            for i in tqdm(range(len(heads)), desc=f"Writing {split_name}"):
                h = heads[i].item()
                r = relations[i].item()
                t = tails[i].item()
                f.write(f"{h}\t{r}\t{t}\n")
        
        print(f"Successfully converted {split_name} split.")

    except Exception as e:
        print(f"An error occurred while processing {split_name}: {e}")

if __name__ == '__main__':
    # Define base paths
    project_root = '/Users/minhbui/Personal/Project/Master/PAPERS/RASG/source/grail'
    input_directory = os.path.join(project_root, 'reference/ogb/examples/linkproppred/biokg/dataset/ogbl_biokg/split/random')
    output_directory = os.path.join(project_root, 'data/ogbl-biokg')

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Convert all three splits
    convert_split('train', input_directory, output_directory)
    convert_split('valid', input_directory, output_directory)
    convert_split('test', input_directory, output_directory)

    print("\nData conversion complete.")
