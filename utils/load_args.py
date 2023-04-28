import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./dataset")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wgts_path', type=str, default="./weights/best_model_py.pth")
    args = parser.parse_args()
    return args