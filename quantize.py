import torch
from torch.utils.data import DataLoader
import numpy as np

from model.util import test, get_dataset, get_dataloaders, get_model, inference_time_cpu, static_quantization, quantization_aware_training, print_size_of_model
from model.models import EarlyStopper


def main(dataset_name,
         dataset_path,
         model_path,
         epochs,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):

    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(dataset, dataset_name, batch_size)
    mini_dataset = torch.utils.data.Subset(dataset, np.arange(512 * 500))
    batch_sizes = [1, 16, 128, 256, 512]

    model = get_model('dfwfm', dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    model.to(torch.device('cpu'))

    # dynamic quantization
    model_dynamic_quantized = torch.quantization.quantize_dynamic(model=model, qconfig_spec={'mlp'}, dtype=torch.qint8)
    loss, auc, prauc, rce = test(model_dynamic_quantized , test_data_loader, criterion, torch.device('cpu'))
    print(f'dynamic quantization test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(model_dynamic_quantized)
    for batch_size in batch_sizes:
        mini_data_loader = torch.utils.data.DataLoader(mini_dataset, batch_size=batch_size, num_workers=0)
        print(f"batch size:\t{batch_size}")
        inference_time_cpu(model_dynamic_quantized , mini_data_loader)

    # static quantization
    model_static_quantized = static_quantization(model, valid_data_loader, criterion)
    loss, auc, prauc, rce = test(model_static_quantized, test_data_loader, criterion, torch.device('cpu'))
    print(f'static quantization test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(model_static_quantized)
    for batch_size in batch_sizes:
        mini_data_loader = torch.utils.data.DataLoader(mini_dataset, batch_size=batch_size, num_workers=0)
        print(f"batch size:\t{batch_size}")
        inference_time_cpu(model_static_quantized , mini_data_loader)

    # QAT
    model_qat = get_model(model_name, dataset, batch_norm=False).to(device) # batch norm not supported in train mode yet
    early_stopper_qat = EarlyStopper(num_trials=2, save_path=f'{model_path[:-3]}_qat.pt')
    model_qat = quantization_aware_training(model_qat, train_data_loader, valid_data_loader, early_stopper_qat, device=device, epochs=epochs)
    loss, auc, prauc, rce = test(model_qat, test_data_loader, criterion, torch.device('cpu'))
    print(f'qat test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(model_qat)
    for batch_size in batch_sizes:
        mini_data_loader = torch.utils.data.DataLoader(mini_dataset, batch_size=batch_size, num_workers=0)
        print(f"batch size:\t{batch_size}")
        inference_time_cpu(model_qat , mini_data_loader)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt', default='G://dac//train_sss.txt')
    parser.add_argument('--model_path', help='path to checkpoint of model, only dfwfm', default='./saved_models/dfwfm.pt')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./saved_models')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_path,
         args.epochs,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)