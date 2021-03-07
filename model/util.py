import torch
from sklearn import metrics
import tqdm
import numpy as np
from time import time, time_ns

from dataset.criteo import CriteoDataset
from dataset.twitter import TwitterDataset
from model.models import FactorizationMachineModel, DeepFactorizationMachineModel, \
    FieldWeightedFactorizationMachineModel, DeepFieldWeightedFactorizationMachineModel


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


def get_model(name, dataset, mlp_dims=(400, 400, 400)):
    field_dims = dataset.field_dims
    if name == 'fm':
        return FactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_lw=False)
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims=field_dims, embed_dim=10, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'fwfm':
        return FieldWeightedFactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_fwlw=False, use_lw=False, use_emb_bag=False, use_qr_emb=False)
    elif name == 'dfwfm':
        return DeepFieldWeightedFactorizationMachineModel(field_dims=field_dims, embed_dim=10, use_fwlw=False, use_lw=False, use_emb_bag=False, use_qr_emb=False, mlp_dims=mlp_dims, dropout=0.2, batch_norm=False)
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

    print('\tAvg forward pass time per batch ({}-Threads)(ms):\t{:.3f}'.format(num_threads, np.mean(time_spent)))
    print('\tAvg forward pass time (batch) ({}-Threads)(ms):\t{:.5f}'.format(num_threads, np.sum(time_spent) / len(data_loader) / data_loader.batch_size))

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

    print('\tAvg forward pass time per batch (GPU)(ms):\t{:.3f}'.format(np.mean(time_spent)))
    print('\tAvg forward pass time (batch) (GPU)(ms):\t{:.5f}'.format(np.sum(time_spent) / len(data_loader) / data_loader.batch_size))
