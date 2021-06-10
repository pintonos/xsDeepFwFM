import torch
from torch.utils.data import DataLoader

from model.util import train_kd, train, test, get_dataset, get_dataloaders, get_model, inference_time_cpu, inference_time_gpu, print_size_of_model
from model.models import EarlyStopper

from util import parameters


def main(args):

    teacher_model_name = 'dfwfm'
    student_model_name = 'mlp'
    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_name, args.dataset_path)
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(dataset, args.dataset_name, args.batch_size)

    teacher_model = get_model(teacher_model_name, dataset).to(device)
    checkpoint = torch.load(args.model_path)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])

    criterion = torch.nn.BCELoss()

    #loss, auc, prauc, rce = test(teacher_model, test_data_loader, criterion, device)
    #print(f'teacher test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

    #grid_dims = [(128, 128, 128), (64, 64, 64), (32, 32, 32), (16, 16)]
    mlp_dims = (200, 200, 200)
    alphas = [0.1, 0.5, 0.9]
    #for mlp_dims in grid_dims:
    for a in alphas:
        print(a)
        student_model = get_model(student_model_name, dataset, mlp_dims=mlp_dims, use_emb_bag=args.use_emb_bag, use_qr_emb=args.use_qr_emb,
                                    qr_collisions=args.qr_collisions, batch_norm=args.use_bn, dropout=args.dropout).to(device)
        optimizer = torch.optim.Adam(params=student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        early_stopper = EarlyStopper(num_trials=2, save_path=f'{args.model_path[:-3]}_kd_{mlp_dims}_alpha_{a}_epochs_{args.epochs}.pt')
        for epoch_i in range(args.epochs):
            train_kd(student_model, teacher_model, optimizer, criterion, train_data_loader, device, alpha=a, temperature=3)
            loss, auc, prauc, rce = test(student_model, valid_data_loader, criterion, device)
            print('epoch:', epoch_i)
            print(f'student valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
            if not early_stopper.is_continuable(student_model, auc, epoch_i, optimizer, loss):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
    
            loss, auc, prauc, rce = test(student_model, test_data_loader, criterion, device)
            print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

        # load best model
        checkpoint = torch.load(f'{args.model_path[:-3]}_kd_{mlp_dims}_alpha_{a}_epochs_{args.epochs}.pt')
        student_model.load_state_dict(checkpoint['model_state_dict'])
        loss, auc, prauc, rce = test(student_model, test_data_loader, criterion, device)
        print(f'{mlp_dims} small model test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
        print_size_of_model(student_model)

    small_model = get_model(student_model_name, dataset, mlp_dims=mlp_dims, batch_norm=args.use_bn, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(params=small_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{args.model_path[:-3]}_small_{mlp_dims}_epochs_{args.epochs}.pt')
    for epoch_i in range(args.epochs):
        train(small_model, optimizer, train_data_loader, criterion, device)
        loss, auc, prauc, rce = test(small_model, valid_data_loader, criterion, device)
        print('epoch:', epoch_i)
        print(f'small model valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
        if not early_stopper.is_continuable(small_model, auc, epoch_i, optimizer, loss):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

        loss, auc, prauc, rce = test(small_model, test_data_loader, criterion, device)
        print(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

    # load best model
    checkpoint = torch.load(f'{args.model_path[:-3]}_small_{mlp_dims}_epochs_{args.epochs}.pt')
    small_model.load_state_dict(checkpoint['model_state_dict'])
    loss, auc, prauc, rce = test(small_model, test_data_loader, criterion, device)
    print(f'{mlp_dims} small model test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
    print_size_of_model(small_model)


if __name__ == '__main__':
    parser = parameters.get_parser()
    args = parser.parse_args()

    main(args)
