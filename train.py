from model import CPM
import argparse

parser = argparse.ArgumentParser("Training options for the CPM model.")

parser.add_argument("--inputs_width", type=int, default=368)
parser.add_argument("--inputs_height", type=int, default=368)
parser.add_argument("--inputs_channel", type=int, default=3)
parser.add_argument("--keypoints_num", type=int, default=10)
parser.add_argument("--stage_num", type=int, default=2)
parser.add_argument("--padding", type=str, default='SAME')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--repeat", type=int, default=100)
parser.add_argument("--lr_decay_rate", type=float, default=0.0001)
parser.add_argument("--save_path", type=str, default='./snapshots/')
parser.add_argument("--training_data_file", type=str, default='./data/train.npz')
parser.add_argument("--warm_up_file", type=str, default='./data/model.h5')

args = parser.parse_args()

Model = CPM(
    inputs_width=args.inputs_width,
    inputs_height=args.inputs_height,
    inputs_channel=args.inputs_channel,
    keypoints_num=args.keypoints_num,
    stage_num=args.stage_num,
    padding=args.padding,
    batch_size=args.batch_size,
    lr=args.lr,
    epochs=args.epochs,
    repeat=args.repeat,
    lr_decay_rate=args.lr_decay_rate,
    save_path=args.save_path,
    training_data_file=args.training_data_file,
    warm_up_file=args.warm_up_file)

if __name__ == "__main__":
    # Model.train()
    Model.keras_train()