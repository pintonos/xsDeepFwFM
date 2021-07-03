import tqdm
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from model.models import EarlyStopper
from model.util import get_dataset, get_dataloaders, get_model, train, test, print_size_of_model, get_full_model_path

from util import parameters
from util.custom_logging import get_logger

def main(args, logger):
    
    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_name, args.dataset_path, args.twitter_label)
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(dataset, args.dataset_name, args.batch_size, random=False)

    criterion = torch.nn.BCELoss()

    epoch = 0
    best_accuracy = 0
    if args.model_path:
        model = get_model(args.model_name, dataset, mlp_dims=args.mlp_dim, dropout=args.dropout, batch_norm=args.use_bn, use_qr_emb=args.use_qr_emb, qr_collisions=args.qr_collisions).to(device)
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_accuracy = checkpoint['accuracy']
    else:
        model = get_model(args.model_name, dataset, mlp_dims=args.mlp_dim, dropout=args.dropout, batch_norm=args.use_bn, use_qr_emb=args.use_qr_emb, qr_collisions=args.qr_collisions).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    logger.info(model)

    save_path = get_full_model_path(save_dir=args.save_dir, dataset_name=args.dataset_name, twitter_label=args.twitter_label, model_name=args.model_name, model=model, epochs=args.epochs + epoch)
    logger.info(save_path)
    early_stopper = EarlyStopper(num_trials=5, save_path=save_path, accuracy=best_accuracy)

    for epoch_i in range(epoch + 1, epoch + args.epochs + 1):
        logger.info(f'epoch: {epoch_i}')
        train(model, optimizer, train_data_loader, criterion, device)

        loss, auc, prauc, rce = test(model, valid_data_loader, criterion, device)
        logger.info(f'valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

        if not early_stopper.is_continuable(model, auc, epoch_i, optimizer, loss):
            logger.info(f'validation: best auc: {early_stopper.best_accuracy}')
            break

        loss, auc, prauc, rce = test(model, test_data_loader, criterion, device)
        logger.info(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    
    # load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss, auc, prauc, rce = test(model, test_data_loader, criterion, device)
    logger.info(f'final test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(model, logger=logger)


if __name__ == '__main__':
    parser = parameters.get_parser()
    args = parser.parse_args()
    logger = get_logger(str(int(time.time())))
    logger.info("FM")
    logger.info(args)
    main(args, logger)