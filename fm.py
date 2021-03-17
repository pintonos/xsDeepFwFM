import tqdm
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import numpy as np

from model.models import EarlyStopper
from model.util import get_dataset, get_dataloaders, get_model, train, test, inference_time_cpu, inference_time_gpu, print_size_of_model


def main(dataset_name,
         dataset_path,
         model_name,
         model_path,
         epochs,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         use_emb_bag,
         use_qr_emb):
         
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(dataset, dataset_name, batch_size)

    if model_path:
        model = get_model(model_name, dataset, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb).to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = get_model(model_name, dataset, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb).to(device)
    print(model)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    early_stopper = EarlyStopper(num_trials=2, save_path=f"{save_dir}/{dataset_name}_{model_name}{model.mlp_dims if model.mlp_dims else ''}{'_emb_bag' if use_emb_bag and not use_qr_emb else ''}{'_qr_emb' if use_qr_emb else ''}.pt")

    for epoch_i in range(epochs):
        train(model, optimizer, train_data_loader, criterion, device)
        loss, auc, prauc, rce = test(model, valid_data_loader, criterion, device)
        print('epoch:', epoch_i)
        print(f'valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
        if not early_stopper.is_continuable(model, auc, epoch_i, optimizer, loss):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

    loss, auc, prauc, rce = test(model, test_data_loader, criterion, device)
    print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt', default='G://dac//train_ssss.txt')
    parser.add_argument('--model_name', help='fm or dfm or fwfm or dfwfm', default='dfwfm')
    parser.add_argument('--model_path', help='path to checkpoint of model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./saved_models')
    parser.add_argument('--use_emb_bag', type=int, default=1)
    parser.add_argument('--use_qr_emb', type=int, default=0)
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.model_path,
         args.epochs,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.use_emb_bag,
         args.use_qr_emb)