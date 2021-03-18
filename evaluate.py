import torch
from torch.utils.data import DataLoader

from model.util import test, get_dataset, get_model, inference_time_cpu, inference_time_gpu, print_size_of_model, get_dataloaders
from model.models import EarlyStopper


def main(dataset_name,
         dataset_path,
         model_name,
         model_path,
         batch_size,
         mlp_dims,
         use_emb_bag,
         use_qr_emb,
         device,
         save_dir):

    device = torch.device(device)
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(dataset_name, dataset_path, batch_size)
    mini_dataset, _ = torch.utils.data.random_split(dataset, (300, len(dataset) - 300))
    mini_data_loader = DataLoader(mini_dataset, batch_size=1, num_workers=0)

    model = get_model(model_name, dataset, mlp_dims=mlp_dims, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = torch.nn.BCELoss()

    loss, auc, prauc, rce = test(model, test_data_loader, criterion, device)
    print(f'small model test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

    inference_time_cpu(model , mini_data_loader, profile=True)
    inference_time_cpu(model , test_data_loader, profile=True)

    inference_time_gpu(model , mini_data_loader, profile=True)
    inference_time_gpu(model , test_data_loader, profile=True)

    print_size_of_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt', default='G://dac//train_sss.txt')
    parser.add_argument('--model_path', help='path to checkpoint of model, only dfwfm', default='./saved_models/dfwfm.pt')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mlp_dims', type=tuple, default=(400,400,400))
    parser.add_argument('--use_emb_bag', type=int, default=1)
    parser.add_argument('--use_qr_emb', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./saved_models')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.model_path,
         args.batch_size,
         args.mlp_dims,
         args.use_emb_bag,
         args.use_qr_emb,
         args.device,
         args.save_dir)