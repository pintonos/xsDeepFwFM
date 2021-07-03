from model.layers import FactorizationMachine
import torch
from sklearn import metrics
import tqdm
import numpy as np
from time import time, time_ns
import os
from torch.utils.data import DataLoader

from dataset.criteo import CriteoDataset
from dataset.twitter import TwitterDataset
from model.models import DeepFactorizationMachineModel, ExtremeDeepFactorizationMachineModel, FieldWeightedFactorizationMachineModel, DeepFieldWeightedFactorizationMachineModel, LogisticRegressionModel, MultiLayerPerceptronModel, FactorizationMachineModel


def compute_prauc(gt, pred):
    prec, recall, thresh = metrics.precision_recall_curve(gt, pred)
    prauc = metrics.auc(recall, prec)
    return prauc


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive / float(len(gt))
    return ctr


def compute_rce(gt, pred):
    cross_entropy = metrics.log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = metrics.log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


def cross_entropy(targets, predictions):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce


def get_dataset(name, path, twitter_label="like"):
    if name == 'twitter':
        return TwitterDataset(path, twitter_label=twitter_label)
    elif name == 'criteo':
        return CriteoDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_datasets(dataset, dataset_name, random=False):
    # twitter dataset is already ordered according to train, valid, test sets
    if dataset_name == 'twitter':
        train_length = 106254462
        valid_length = 9760684
        test_length = 9765321

    # criteo split first 6 days in training and last for valid+testing
    else:
        train_length = int(len(dataset) * 0.85)
        valid_length = int(len(dataset) * 0.075)
        test_length = len(dataset) - train_length - valid_length
    if random:
        return torch.utils.data.random_split(dataset, (train_length, valid_length, test_length), generator=torch.Generator().manual_seed(42)) 

    train_indices = np.arange(train_length)
    valid_indices = np.arange(train_length, train_length+valid_length)
    test_indices = np.arange(train_length + valid_length, len(dataset))

    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, valid_indices), torch.utils.data.Subset(dataset, test_indices)


def get_dataloaders(dataset, dataset_name, batch_size, random=False):
    train_dataset, valid_dataset, test_dataset = get_datasets(dataset, dataset_name, random)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    return train_data_loader, valid_data_loader, test_data_loader


def get_model(name, dataset, mlp_dims=(400, 400, 400), dropout=0.0, batch_norm=True, use_qr_emb=False, qr_collisions=4, return_raw_logits=False):
    field_dims = dataset.field_dims
    if type(mlp_dims) is int:
        mlp_dims = (mlp_dims, mlp_dims, mlp_dims)
    if name == 'fwfm' or mlp_dims == (0,0,0):
        return FieldWeightedFactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_fwlw=True, use_lw=False, use_qr_emb=use_qr_emb)
    elif name == 'dfwfm':
        return DeepFieldWeightedFactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_fwlw=True, use_lw=False, use_qr_emb=use_qr_emb, qr_collisions=qr_collisions,
                                                          mlp_dims=mlp_dims, dropout=dropout, batch_norm=batch_norm, return_raw_logits=return_raw_logits)
    elif name == 'mlp':
        return MultiLayerPerceptronModel(field_dims=field_dims, embed_dim=10, mlp_dims=mlp_dims, dropout=dropout, batch_norm=batch_norm, return_raw_logits=return_raw_logits)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_qr_emb=use_qr_emb, qr_collisions=qr_collisions)
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims=field_dims, embed_dim=10, mlp_dims=mlp_dims)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(field_dims=field_dims, embed_dim=10, mlp_dims=mlp_dims, dropout=dropout, cross_layer_sizes=(200, 200, 200))
    elif name == 'lr':
        return LogisticRegressionModel(field_dims=field_dims)
    else:
        raise ValueError('unknown model name: ' + name)


def get_full_model_path(save_dir, dataset_name, twitter_label, model_name, model, epochs):
    return f"{save_dir}/{dataset_name if dataset_name != 'twitter' else dataset_name + '_' + twitter_label}_{model_name}{model.mlp_dims if hasattr(model, 'mlp_dims') else ''}{'_qr' + str(model.qr_collisions) if hasattr(model, 'use_qr_emb') and model.use_qr_emb else ''}_{epochs}.pt"


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def train_kd(student_model, teacher_model, optimizer, criterion, data_loader, device, alpha=0.9, temperature=3, log_interval=100):
    student_model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)

        output_student = student_model(fields)
        output_teacher = teacher_model(fields)

        loss = loss_fn_kd(output_student, output_teacher, target.float(), alpha, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def loss_fn_kd(student_logits, teacher_logits, y, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    Using binary cross entropy instead of cross entropy and sigmoid instead of softmax, since it is a binary classification problem.
    """

    soft_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(student_logits / temperature),
                                    torch.sigmoid(teacher_logits / temperature).detach())

    hard_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(student_logits), y)

    return (alpha * (temperature ** 2) * soft_loss) + ((1. - alpha) * hard_loss)


def test(model, data_loader, criterion, device):
    model.eval()
    model.to(device)
    targets, predicts = list(), list()
    total_loss = 0
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = criterion(y, target.float())
            total_loss += loss.item()

            predicts.extend(y.cpu().data.numpy().astype("float64"))
            targets.extend(target.cpu().data.numpy().astype("float64"))

    return total_loss / len(data_loader), metrics.roc_auc_score(targets, predicts), compute_prauc(targets, predicts), compute_rce(targets, predicts)


def profile_inference(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            fields, target = fields.to(device), target.to(device)
            if len(data_loader) < i < 100:  # warmup
                y = model(fields)
            else:
                with torch.autograd.profiler.profile(with_stack=True, use_cuda=True if device != 'cpu' else False) as prof:
                    y = model(fields)

    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))
    #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))


def inference_time_cpu(model, data_loader, num_threads=1, profile=False):
    device = 'cpu'
    model.to(device)
    torch.set_num_threads(num_threads)
    model.eval()

    if profile:
        profile_inference(model, data_loader, device)
    
    time_spent = []
    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            fields, target = fields.to(device), target.to(device)
            if len(data_loader) < i < 100:  # warmup
                _ = model(fields)
            else:
                start_time = time_ns()
                _ = model(fields)
                time_on_batch_ms = (time_ns() - start_time) // 1_000_000
                time_spent.append(time_on_batch_ms)

    print('\tAvg time per batch ({}-Threads)(ms):\t{:.3f}'.format(num_threads, np.mean(time_spent)))
    print('\tAvg time per item ({}-Threads)(ms):\t{:.5f}'.format(num_threads, np.mean(time_spent) / data_loader.batch_size))


def inference_time_gpu(model, data_loader, profile=False):
    device = 'cuda:0'
    model.to(device)
    model.eval()

    if profile:
        profile_inference(model, data_loader, device)

    time_spent = []
    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            fields, target = fields.to(device), target.to(device)
            if len(data_loader) < i < 100:  # warmup
                y = model(fields)
            else:
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(fields)
                end.record()
                torch.cuda.synchronize()
                time_on_batch_ms = start.elapsed_time(end)
                time_spent.append(time_on_batch_ms)

    print('\tAvg time per batch (GPU)(ms):\t{:.3f}'.format(np.mean(time_spent)))
    print('\tAvg time per item (GPU)(ms):\t{:.5f}'.format(np.mean(time_spent) / data_loader.batch_size))


def print_size_of_model(model, logger=None):
    print('========')
    print('MODEL SIZE')
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    os.remove('temp.p')

    num_total = 0
    for _, param in model.named_parameters():
        num_total += np.prod(param.data.shape)

    if logger:
        logger.info('\tSize (MB):\t' + str(size / 1e6))
        logger.info(f"\tNumber of Parameters: \t{num_total:,}")
    else:
        print('\tSize (MB):\t' + str(size / 1e6))
        print(f"\tNumber of Parameters: \t{num_total:,}")


def save_model(model, save_path, epoch=1, optimizer=None, loss=None):
    if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)


def static_quantization(model, dataloader, criterion, dropout_layer=False):
    device = torch.device('cpu')
    model.eval()
    model.to(device)
    model.mlp.quantize = True
    model.mlp.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model.embeddings.qconfig = torch.quantization.float_qparams_weight_only_qconfig

    layer_offset = 4 if dropout_layer else 3
    layer_start = 1 if dropout_layer else 0

    #  fuse linear and batchnorm1d
    model_fused = torch.quantization.fuse_modules(model,
                                                          [['mlp.mlp.' + str(layer_start),
                                                            'mlp.mlp.' + str(layer_start + 1)],
                                                           ['mlp.mlp.' + str(layer_offset),
                                                            'mlp.mlp.' + str((layer_offset + 1))],
                                                           ['mlp.mlp.' + str(layer_offset * 2),
                                                            'mlp.mlp.' + str((layer_offset * 2 + 1))]])

    #  fuse linear and relu
    model_fused = torch.quantization.fuse_modules(model_fused,
                                                          [['mlp.mlp.' + str(layer_start),
                                                            'mlp.mlp.' + str(layer_start + 2)],
                                                           ['mlp.mlp.' + str(layer_offset),
                                                            'mlp.mlp.' + str((layer_offset + 2))],
                                                           ['mlp.mlp.' + str(layer_offset * 2),
                                                            'mlp.mlp.' + str((layer_offset * 2 + 2))]])
    model_prepared = torch.quantization.prepare(model_fused)
    _, _, _, _ = test(model_prepared, dataloader, criterion, device)  # calibrate

    model_static_quantized = torch.quantization.convert(model_prepared)
    return model_static_quantized


def quantization_aware_training(model, train_data_loader, valid_data_loader, test_data_loader, early_stopper, device, logger, epochs=50, model_path=None):
    model.train()
    model.mlp.quantize = True
    model.mlp.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model.embeddings.qconfig = torch.quantization.float_qparams_weight_only_qconfig
    
    #  fusing linear and batchnorm1d in training not supported yet
    #  fuse linear and relu
    #  use fixed layer numbering!
    model_fused = torch.quantization.fuse_modules(model,
                                                [['mlp.mlp.1', 'mlp.mlp.2'],
                                                ['mlp.mlp.4', 'mlp.mlp.5'],
                                                ['mlp.mlp.7', 'mlp.mlp.8']])

    model_prepared = torch.quantization.prepare_qat(model_fused)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model_prepared.parameters(), lr=0.001, weight_decay=1e-6)

    epoch = 0
    if model_path:
        checkpoint = torch.load(model_path)
        model_prepared.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        early_stopper.best_accuracy = checkpoint['accuracy']
    for epoch_i in range(epoch + 1, epoch + epochs + 1):
        train(model_prepared, optimizer, train_data_loader, criterion, device)
        loss, auc, prauc, rce = test(model_prepared, valid_data_loader, criterion, device)
        logger.info(f'epoch: {epoch_i}')
        logger.info(f'valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
        if not early_stopper.is_continuable(model_prepared, auc, epoch_i, optimizer, loss):
            logger.info(f'validation: best auc: {early_stopper.best_accuracy}')
            break
        
        loss, auc, prauc, rce = test(model_prepared, test_data_loader, criterion, device)
        logger.info(f'test loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')

    model_prepared.eval()
    model_prepared.to(torch.device('cpu'))
    model_qat = torch.quantization.convert(model_prepared)
    
    return model_qat