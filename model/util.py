import torch
from sklearn import metrics
import tqdm
import numpy as np
from time import time, time_ns
import os

from dataset.criteo import CriteoDataset
from dataset.twitter import TwitterDataset
from model.models import FactorizationMachineModel, DeepFactorizationMachineModel, \
    FieldWeightedFactorizationMachineModel, DeepFieldWeightedFactorizationMachineModel, MultiLayerPerceptronModel


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


def get_dataset(name, path):
    if name == 'twitter':
        return TwitterDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset, mlp_dims=(400, 400, 400), batch_norm=True):
    field_dims = dataset.field_dims
    if name == 'fm':
        return FactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_lw=False)
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims=field_dims, embed_dim=10, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'fwfm':
        return FieldWeightedFactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_fwlw=True, use_lw=False, use_emb_bag=False, use_qr_emb=False)
    elif name == 'dfwfm':
        return DeepFieldWeightedFactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_fwlw=True, use_lw=False, use_emb_bag=False, use_qr_emb=False, mlp_dims=mlp_dims, dropout=0.2, batch_norm=batch_norm)
    elif name == 'mlp':
        return MultiLayerPerceptronModel(field_dims=field_dims, embed_dim=10, mlp_dims=mlp_dims, dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)


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


def train_kd(student_model, teacher_model, optimizer, data_loader, device, alpha=0.9, temperature=3, log_interval=100):
    criterion = torch.nn.BCELoss()

    student_model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)

        output_student = student_model(fields)
        output_teacher = teacher_model(fields)

        loss = loss_fn_kd(output_student, output_teacher, target.float(), alpha, temperature)
        loss = criterion(output_student, target.float())

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def loss_fn_kd(outputs, teacher_outputs, y, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
    """
    kd_loss = torch.nn.KLDivLoss()(torch.nn.functional.log_softmax(outputs / temperature, dim=0),
                                torch.nn.functional.softmax(teacher_outputs / temperature, dim=0)) * (alpha * temperature * temperature) + \
                torch.nn.functional.binary_cross_entropy_with_logits(outputs, y) * (1. - alpha)

    return kd_loss


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
                with torch.autograd.profiler.profile(with_stack=True, use_cuda=False) as prof:
                    y = model(fields)

    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=5))
    #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=20))


def inference_time_cpu(model, data_loader, num_threads=1):
    device = 'cpu'
    model.to(device)
    torch.set_num_threads(num_threads)
    model.eval()

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


def inference_time_gpu(model, data_loader):
    device = 'cuda:0'
    model.to(device)
    model.eval()

    profile_inference(model, data_loader, device)

    time_spent = []
    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            fields, target = fields.to(device), target.to(device)
            if len(data_loader) < i < 100:  # warmup
                y = model(fields)
            else:
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


def print_size_of_model(model):
    print('========')
    print('MODEL SIZE')
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print('\tSize (MB):\t' + str(size / 1e6))
    os.remove('temp.p')

    num_total = 0
    for name, param in model.named_parameters():
        num_total += np.prod(param.data.shape)
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


def static_quantization(model, dataloader, criterion):
    device = torch.device('cpu')
    model.eval()
    model.to(device)
    model.mlp.quantize = True
    model.mlp.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model.fwfm.embeddings.qconfig = torch.quantization.float_qparams_weight_only_qconfig

    #  fuse linear and batchnorm1d
    model_fused = torch.quantization.fuse_modules(model,
                                                          [['mlp.mlp.0', 'mlp.mlp.1'],
                                                           ['mlp.mlp.4', 'mlp.mlp.5'],
                                                           ['mlp.mlp.8', 'mlp.mlp.9']])
    #  fuse linear and relu
    model_fused = torch.quantization.fuse_modules(model_fused,
                                                          [['mlp.mlp.0', 'mlp.mlp.2'],
                                                           ['mlp.mlp.4', 'mlp.mlp.6'],
                                                           ['mlp.mlp.8', 'mlp.mlp.10']])

    model_prepared = torch.quantization.prepare(model_fused)
    _, _, _, _ = test(model_prepared, dataloader, criterion, device)  # calibrate

    model_static_quantized = torch.quantization.convert(model_prepared)
    return model_static_quantized

def quantization_aware_training(model, train_data_loader, valid_data_loader, early_stopper, device, epochs=3):
    model.train()
    model.mlp.quantize = True
    model.mlp.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model.fwfm.embeddings.qconfig = torch.quantization.float_qparams_weight_only_qconfig

    #  fusing linear and batchnorm1d in training not supported yet
    #  fuse linear and relu
    model_fused = torch.quantization.fuse_modules(model,
                                                [['mlp.mlp.0', 'mlp.mlp.1'],
                                                ['mlp.mlp.3', 'mlp.mlp.4'],
                                                ['mlp.mlp.6', 'mlp.mlp.7']])

    model_prepared = torch.quantization.prepare_qat(model_fused)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model_prepared.parameters(), lr=0.001, weight_decay=1e-6)
    for epoch_i in range(epochs):
        if epoch_i > epochs - 2:
            # Freeze quantizer parameters
            model_prepared.apply(torch.quantization.disable_observer)
        if epoch_i > epochs - 1:
            # Freeze batch norm mean and variance estimates
            model_prepared.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        train(model_prepared, optimizer, train_data_loader, criterion, device)
        loss, auc, prauc, rce = test(model_prepared, valid_data_loader, criterion, device)
        print('epoch:', epoch_i)
        print(f'valid loss: {loss:.6f} auc: {auc:.6f} prauc: {prauc:.4f} rce: {rce:.4f}')
        if not early_stopper.is_continuable(model_prepared, auc, epoch_i, optimizer, loss):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

    model_prepared.eval()
    model_prepared.to(torch.device('cpu'))
    model_qat = torch.quantization.convert(model_prepared)
    
    return model_qat