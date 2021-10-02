import os
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, help="path to generated_data.pt")
    parser.add_argument("-s", default="../nuisance-orthogonal-prediction/code/nrd-xray/erm-on-generated/", type=str, help="save dir")
    args = parser.parse_args()

    data = torch.load(os.path.join(args.n))
    dataset = TensorDataset(torch.cat([batch[0] for batch in data], axis=0), torch.cat([batch[1] for batch in data], axis=0), torch.cat([batch[2] for batch in data], axis=0))
    torch.save(dataset, os.path.join(args.s, "generated_data1.pt"))    



