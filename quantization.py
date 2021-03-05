import torch
from torch.utils.data import DataLoader

from model.util import test, get_dataset, get_model, inference_time
from model.models import EarlyStopper


def main(dataset_name,
         dataset_path,
         model_name,
         model_path,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):

    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=8192, num_workers=0)

    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    quantized_model = model.eval()
    device = torch.device('cpu')
    quantized_model.to(device)

    quantized_model.qconfig = torch.quantization.default_qconfig#torch.quantization.get_default_qconfig('fbgemm')
    quantized_model.fwfm.embeddings.qconfig = None #torch.quantization.float_qparams_weight_only_qconfig
    quantized_model.linear.qconfig = None
    quantized_model.fwfm.field_cov.qconfig = None
    quantized_model.fwfm_linear.qconfig = None

    # TODO cannot fuse linear and relu because batch norm in between
    # https: // github.com / PyTorchLightning / pytorch - lightning / issues / 2544
    # https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989/9 where to put batchnorm
    #quantized_model = torch.quantization.fuse_modules(quantized_model,
                                                      #[['mlp.mlp.0', 'mlp.mlp.1', 'mlp.mlp.2'],
                                                      # ['mlp.mlp.4', 'mlp.mlp.5', 'mlp.mlp.6'],
                                                      # ['mlp.mlp.8', 'mlp.mlp.9', 'mlp.mlp.10']])
    quantized_model = torch.quantization.fuse_modules(quantized_model,
                                                          [['mlp.mlp.0', 'mlp.mlp.1'],
                                                           ['mlp.mlp.3', 'mlp.mlp.4'],
                                                           ['mlp.mlp.6', 'mlp.mlp.7']])

    torch.quantization.prepare(quantized_model, inplace=True)
    _, _, _, _ = test(quantized_model, valid_data_loader, criterion, device)  # calibrate

    torch.quantization.convert(quantized_model, inplace=True)

    quantized_model.mlp.quantize = True
    loss, auc, prauc, rce = test(quantized_model, test_data_loader, criterion, device)
    print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

    mini_dataset, _ = torch.utils.data.random_split(dataset, (300, len(dataset) - 300))
    mini_data_loader = DataLoader(mini_dataset, batch_size=1, num_workers=0)
    inference_time(model, mini_data_loader, torch.device('cpu'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt', default='G://dac//train_ssss.txt')
    parser.add_argument('--model_name', help='dfm or dfwfm', default='dwfwfm')
    parser.add_argument('--model_path', help='path to checkpoint of model', default='./saved_models/dwfwfm.pt')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./saved_models')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.model_path,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
