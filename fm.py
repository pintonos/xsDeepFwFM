import tqdm
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import numpy as np
import os

from model.models import EarlyStopper
from model.util import get_dataset, get_dataloaders, get_model, train, test, inference_time_cpu, inference_time_gpu, print_size_of_model, get_full_model_path


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
         use_qr_emb,
         qr_collisions,
         twitter_label,
         dropout,
         use_batch_norm):
    
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path, twitter_label)
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(dataset, dataset_name, batch_size, random=False)

    criterion = torch.nn.BCELoss()

    epoch = 0
    if model_path:
        model = get_model(model_name, dataset, dropout=dropout, batch_norm=use_batch_norm, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb, qr_collisions=qr_collisions).to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        model = get_model(model_name, dataset, dropout=dropout, batch_norm=use_batch_norm, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb, qr_collisions=qr_collisions).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(model)

    save_path = get_full_model_path(save_dir, dataset_name, twitter_label, model_name, model, use_emb_bag, use_qr_emb, qr_collisions, epochs + epoch)
    early_stopper = EarlyStopper(num_trials=2, save_path=save_path)

    for epoch_i in range(epoch + 1, epoch + epochs + 1):
        print('epoch:', epoch_i)
        train(model, optimizer, train_data_loader, criterion, device)

        loss, auc, prauc, rce = test(model, valid_data_loader, criterion, device)
        print(f'valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

        if not early_stopper.is_continuable(model, auc, epoch_i, optimizer, loss):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

        loss, auc, prauc, rce = test(model, test_data_loader, criterion, device)
        print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    
    # load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss, auc, prauc, rce = test(model, test_data_loader, criterion, device)
    print(f'final test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt', default='./data/criteo/train.txt')
    parser.add_argument('--model_name', help='fwfm or dfwfm', default='dfwfm')
    parser.add_argument('--model_path', help='path to checkpoint of model')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', help='L2', type=float, default=1e-6) # default: 1e-6, xdeepfm: 0.0001, deepfwfm: 3e-7, dcn: 0.0, nfm: 0.4 dropout or 1e-4
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./saved_models')
    parser.add_argument('--use_emb_bag', type=int, default=1)
    parser.add_argument('--use_qr_emb', type=int, default=0)
    parser.add_argument('--qr_collisions', type=int, default=4)
    parser.add_argument('--twitter_label', default='like')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_bn', help='use batch norm', type=int, default=1)
    args = parser.parse_args()
    print(args)
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
         args.use_qr_emb,
         args.qr_collisions,
         args.twitter_label,
         args.dropout,
         args.use_bn)