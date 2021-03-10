import torch
from torch.utils.data import DataLoader

from model.util import test, get_dataset, get_model, inference_time_cpu
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
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

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

    # dynamic quantization
    model_dynamic_quantized  = torch.quantization.quantize_dynamic(model=quantized_model, qconfig_spec={'mlp'}, dtype=torch.qint8)

    loss, auc, prauc, rce = test(model_dynamic_quantized , test_data_loader, criterion, device)
    print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    inference_time_cpu(model_dynamic_quantized , test_data_loader)

    # static quantization
    quantized_model.mlp.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    quantized_model.fwfm.embeddings.qconfig = torch.quantization.float_qparams_weight_only_qconfig

    #  fuse linear and batchnorm1d
    quantized_model = torch.quantization.fuse_modules(quantized_model,
                                                          [['mlp.mlp.0', 'mlp.mlp.1'],
                                                           ['mlp.mlp.4', 'mlp.mlp.5'],
                                                           ['mlp.mlp.8', 'mlp.mlp.9']])
    #  fuse linear and relu
    quantized_model = torch.quantization.fuse_modules(quantized_model,
                                                          [['mlp.mlp.0', 'mlp.mlp.2'],
                                                           ['mlp.mlp.4', 'mlp.mlp.6'],
                                                           ['mlp.mlp.8', 'mlp.mlp.10']])

    torch.quantization.prepare(quantized_model, inplace=True)
    print(quantized_model)
    _, _, _, _ = test(quantized_model, valid_data_loader, criterion, device)  # calibrate

    model_static_quantized = torch.quantization.convert(quantized_model)

    model_static_quantized.mlp.quantize = True
    loss, auc, prauc, rce = test(model_static_quantized, test_data_loader, criterion, device)
    print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    inference_time_cpu(model_static_quantized, test_data_loader)

    # original model
    loss, auc, prauc, rce = test(model, test_data_loader, criterion, device)
    print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    inference_time_cpu(model, test_data_loader)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt', default='G://dac//train_sss.txt')
    parser.add_argument('--model_name', help='dfm or dfwfm', default='dfwfm')
    parser.add_argument('--model_path', help='path to checkpoint of model', default='./saved_models/dfwfm.pt')
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