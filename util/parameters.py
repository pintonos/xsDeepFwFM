import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt', default='./data/criteo/train.txt')
    parser.add_argument('--model_name', help='fwfm or dfwfm', default='dfwfm')
    parser.add_argument('--model_path', help='path to checkpoint of model')
    parser.add_argument('--teacher_model_path', help='path to checkpoint of teacher model')
    parser.add_argument('--mlp_dim', help='neurons per layer', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', help='L2', type=float, default=1e-6) # default: 1e-6, xdeepfm: 0.0001, deepfwfm: 3e-7, dcn: 0.0, nfm: 0.4 dropout or 1e-4
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./saved_models')
    parser.add_argument('--use_emb_bag', type=int, default=1)
    parser.add_argument('--use_qr_emb', type=int, default=0)
    parser.add_argument('--qr_collisions', type=int, default=4)
    parser.add_argument('--twitter_label', default='like')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_bn', help='use batch norm', type=int, default=1)
    parser.add_argument('--profile_inference', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--temperature', type=int, default=3)

    return parser