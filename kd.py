import torch
from torch.utils.data import DataLoader

from model.util import train_kd, test, get_dataset, get_model, inference_time_cpu, inference_time_gpu
from model.models import EarlyStopper


def main(dataset_name,
         dataset_path,
         epochs,
         model_name,
         model_path,
         learning_rate,
         batch_size,
         weight_decay,
         alpha,
         temperature,
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

    teacher_model = get_model(model_name, dataset).to(device)
    checkpoint = torch.load(model_path)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])

    criterion = torch.nn.BCELoss()

    loss, auc, prauc, rce = test(teacher_model, test_data_loader, criterion, device)
    print(f'teacher test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

    student_model = get_model(model_name, dataset, mlp_dims=(100, 100, 100)).to(device)
    optimizer = torch.optim.Adam(params=student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}_kd.pt')

    for epoch_i in range(epochs):
        train_kd(student_model, teacher_model, optimizer, train_data_loader, device, alpha=alpha, temperature=temperature)
        loss, auc, prauc, rce = test(student_model, valid_data_loader, criterion, device)
        print('epoch:', epoch_i)
        print(f'valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
        if not early_stopper.is_continuable(student_model, auc, epoch_i, optimizer, loss):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

    inference_time_cpu(teacher_model, test_data_loader)
    inference_time_cpu(student_model, test_data_loader)

    inference_time_gpu(teacher_model, test_data_loader)
    inference_time_gpu(student_model, test_data_loader)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt', default='G://dac//train_ssss.txt')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_name', help='dfm or dfwfm', default='dfwfm')
    parser.add_argument('--model_path', help='path to checkpoint of model', default='./saved_models/dfwfm.pt')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=2.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./saved_models')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.epochs,
         args.model_name,
         args.model_path,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.alpha,
         args.temperature,
         args.device,
         args.save_dir)
