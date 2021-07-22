import torch
from torch.utils.data import DataLoader

from model.util import train_kd, test, get_dataset, get_dataloaders, get_model, print_size_of_model
from model.models import EarlyStopper

from util import parameters

from util.custom_logging import get_logger
import time


def main(args, logger):

    teacher_model_name = 'dfwfm'
    student_model_name = 'mlp'
    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_name, args.dataset_path, twitter_label=args.twitter_label)
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(dataset, args.dataset_name, args.batch_size)

    teacher_model = get_model(teacher_model_name, dataset, batch_norm=args.use_bn, dropout=args.dropout, return_raw_logits=True).to(device)
    checkpoint = torch.load(args.base_model_path)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])

    criterion = torch.nn.BCELoss()

    # load model
    epoch = 0
    best_accuracy = 0
    if args.model_path:
        student_model = get_model(student_model_name, dataset, mlp_dims=args.mlp_dim, use_qr_emb=args.use_qr_emb,
                                qr_collisions=args.qr_collisions, batch_norm=args.use_bn, dropout=args.dropout, return_raw_logits=True).to(device)
        checkpoint = torch.load(args.model_path)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(params=student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_accuracy = checkpoint['accuracy']
    else:
        student_model = get_model(student_model_name, dataset, mlp_dims=args.mlp_dim, use_qr_emb=args.use_qr_emb,
                                qr_collisions=args.qr_collisions, batch_norm=args.use_bn, dropout=args.dropout, return_raw_logits=True).to(device)
        optimizer = torch.optim.Adam(params=student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    logger.info(student_model)

    # train model with KD
    save_path = f'{args.base_model_path[:-3]}_kd_{args.mlp_dim}_alpha_{args.alpha}_qr_2.pt'
    logger.info(save_path)
    early_stopper = EarlyStopper(num_trials=5, save_path=save_path, accuracy=best_accuracy)
    for epoch_i in range(epoch + 1, epoch + args.epochs + 1):
        train_kd(student_model, teacher_model, optimizer, criterion, train_data_loader, device, alpha=args.alpha, temperature=args.temperature)
        student_model.return_raw_logits = False
        loss, auc, prauc, rce = test(student_model, valid_data_loader, criterion, device)
        logger.info(f'epoch: {epoch_i}')
        logger.info(f'student valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
        if not early_stopper.is_continuable(student_model, auc, epoch_i, optimizer, loss):
            logger.info(f'validation: best auc: {early_stopper.best_accuracy}')
            break

        loss, auc, prauc, rce = test(student_model, test_data_loader, criterion, device)
        logger.info(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
        student_model.return_raw_logits = True

    # load best model
    checkpoint = torch.load(save_path)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.return_raw_logits = False
    loss, auc, prauc, rce = test(student_model, test_data_loader, criterion, device)
    logger.info(f'{args.mlp_dim} small model test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(student_model)


if __name__ == '__main__':
    parser = parameters.get_parser()
    args = parser.parse_args()
    logger = get_logger(str(int(time.time())))
    logger.info("KD")
    logger.info(args)
    main(args, logger)
