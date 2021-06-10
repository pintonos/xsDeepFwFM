import torch
import numpy as np

from model.util import test, get_dataset, get_datasets, get_model, inference_time_cpu, inference_time_gpu, print_size_of_model, get_dataloaders
from model.models import EarlyStopper


def main(dataset_name,
         dataset_path,
         model_name,
         model_path,
         batch_size,
         mlp_dims,
         use_emb_bag,
         use_qr_emb,
         qr_collisions,
         device,
         profile,
         twitter_label):

    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path, twitter_label)
    _, _, test_dataset = get_datasets(dataset, dataset_name)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    model = get_model(model_name, dataset, mlp_dims=mlp_dims, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb, qr_collisions=qr_collisions).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = torch.nn.BCELoss()

    loss, auc, prauc, rce = test(model, test_data_loader, criterion, device)
    print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

    print_size_of_model(model)

    # CPU
    batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024]
    for batch_size in batch_sizes:
        batched_dataset = torch.utils.data.Subset(dataset, np.arange(batch_size * 500))
        batched_data_loader = torch.utils.data.DataLoader(batched_dataset, batch_size=batch_size, num_workers=0)
        print(f"batch size:\t{batch_size}")
        inference_time_cpu(model, batched_data_loader, profile=profile)
    
    # GPU
    batch_sizes = [512, 1024, 2048, 4096]
    for batch_size in batch_sizes:
        batched_dataset = torch.utils.data.Subset(dataset, np.arange(batch_size * 500))
        batched_data_loader = torch.utils.data.DataLoader(batched_dataset, batch_size=batch_size, num_workers=0)
        print(f"batch size:\t{batch_size}")
        inference_time_gpu(model, batched_data_loader, profile=profile)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt', default='./data/criteo/train.txt')
    parser.add_argument('--model_name', help='fwfm or dfwfm', default='dfwfm')
    parser.add_argument('--model_path', help='path to checkpoint of model, only dfwfm', default='./saved_models/criteo_dfwfm(400, 400, 400)_emb_bag.pt')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mlp_dims', type=tuple, default=(400,400,400))
    parser.add_argument('--use_emb_bag', type=int, default=1)
    parser.add_argument('--use_qr_emb', type=int, default=0)
    parser.add_argument('--qr_collisions', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--profile', type=int, default=1)
    parser.add_argument('--twitter_label', default='like')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.model_path,
         args.batch_size,
         args.mlp_dims,
         args.use_emb_bag,
         args.use_qr_emb,
         args.qr_collisions,
         args.device,
         args.profile,
         args.twitter_label)