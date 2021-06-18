import torch
import numpy as np

from model.util import test, get_dataset, get_datasets, get_model, inference_time_cpu, inference_time_gpu, print_size_of_model

from util import parameters

def main(args):

    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_name, args.dataset_path, args.twitter_label)
    _, _, test_dataset = get_datasets(dataset, args.dataset_name)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    model = get_model(args.model_name, dataset, mlp_dims=args.mlp_dims, use_emb_bag=args.use_emb_bag, use_qr_emb=args.use_qr_emb, qr_collisions=args.qr_collisions).to(device)
    print(model)
    checkpoint = torch.load(args.model_path)
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
        inference_time_cpu(model, batched_data_loader, profile=args.profile_inference)
    
    # GPU
    batch_sizes = [512, 1024, 2048, 4096]
    for batch_size in batch_sizes:
        batched_dataset = torch.utils.data.Subset(dataset, np.arange(batch_size * 500))
        batched_data_loader = torch.utils.data.DataLoader(batched_dataset, batch_size=batch_size, num_workers=0)
        print(f"batch size:\t{batch_size}")
        inference_time_gpu(model, batched_data_loader, profile=args.profile_inference)


if __name__ == '__main__':
    parser = parameters.get_parser()
    args = parser.parse_args()

    main(args)