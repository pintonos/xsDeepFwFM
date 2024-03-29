import torch
from torch.utils.data import DataLoader
import numpy as np

from model.util import test, get_dataset, get_dataloaders, get_model, inference_time_cpu, static_quantization, quantization_aware_training, print_size_of_model
from model.models import EarlyStopper

from util import parameters
from util.custom_logging import get_logger
import time


def main(args, logger):

    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_name, args.dataset_path, args.twitter_label)
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(dataset, args.dataset_name, args.batch_size)
    mini_dataset = torch.utils.data.Subset(dataset, np.arange(1024 * 500))
    batch_sizes = [1, 64, 128, 256, 512]

    model = get_model(args.model_name, dataset, mlp_dims=args.mlp_dim, dropout=args.dropout, batch_norm=args.use_bn, use_qr_emb=args.use_qr_emb, qr_collisions=args.qr_collisions).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    checkpoint = torch.load(args.base_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    model.to(torch.device('cpu'))
    logger.info(model)

    # dynamic quantization
    '''model_dynamic_quantized = torch.quantization.quantize_dynamic(model=model, qconfig_spec={'mlp'}, dtype=torch.qint8)
    loss, auc, prauc, rce = test(model_dynamic_quantized , test_data_loader, criterion, torch.device('cpu'))
    logger.info(f'dynamic quantization test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(model_dynamic_quantized)
    for batch_size in batch_sizes:
        batched_dataset = torch.utils.data.Subset(dataset, np.arange(batch_size * 500))
        batched_data_loader = torch.utils.data.DataLoader(batched_dataset, batch_size=batch_size, num_workers=0)
        logger.info(f"batch size:\t{batch_size}")
        inference_time_cpu(model_dynamic_quantized, batched_data_loader, profile=args.profile_inference)'''

    # static quantization
    model_static_quantized = static_quantization(model, valid_data_loader, criterion, dropout_layer=True if args.dropout > 0.0 else False)
    loss, auc, prauc, rce = test(model_static_quantized, test_data_loader, criterion, torch.device('cpu'))
    logger.info(f'static quantization test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(model_static_quantized)
    for batch_size in batch_sizes:
        batched_dataset = torch.utils.data.Subset(dataset, np.arange(batch_size * 500))
        batched_data_loader = torch.utils.data.DataLoader(batched_dataset, batch_size=batch_size, num_workers=0)
        logger.info(f"batch size:\t{batch_size}")
        inference_time_cpu(model_static_quantized, batched_data_loader, profile=args.profile_inference)

    # QAT
    '''
    QAT does not support fusing of linear + batchnorm1d and run QuantizedCPU with batchnorm1d.
    -> use dropout instead
    '''
    '''model_qat = get_model('dfwfm', dataset, batch_norm=False, dropout=0.2).to(device) # dropout is fixed here!
    logger.info(model_qat)
    early_stopper_qat = EarlyStopper(num_trials=2, save_path=f'{args.base_model_path[:-3]}_qat.pt')
    model_qat = quantization_aware_training(model_qat, train_data_loader, valid_data_loader, test_data_loader, early_stopper_qat, device=device, logger=logger, epochs=args.epochs, model_path=args.model_path)

    loss, auc, prauc, rce = test(model_qat, test_data_loader, criterion, torch.device('cpu'))
    logger.info(f'qat test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(model_qat)

    for batch_size in batch_sizes:
        batched_dataset = torch.utils.data.Subset(dataset, np.arange(batch_size * 500))
        batched_data_loader = torch.utils.data.DataLoader(batched_dataset, batch_size=batch_size, num_workers=0)
        logger.info(f"batch size:\t{batch_size}")
        inference_time_cpu(model_qat, batched_data_loader, profile=args.profile_inference)'''


if __name__ == '__main__':
    parser = parameters.get_parser()
    args = parser.parse_args()
    logger = get_logger(str(int(time.time())))
    logger.info("Quantization")
    logger.info(args)
    main(args, logger)