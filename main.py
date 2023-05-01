"""To train a simple localization model, one sample fit to test CI/CD Github Actions."""
import torch
import torchvision
import logging
import albumentations as A

from train.fit import fit
from model.CNN import Model
from utils.load_args import get_args
from utils.data import get_sample, get_aug, norm, load_model

logging.basicConfig(
    filename="runing.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
)

if __name__ == "__main__":
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    classes = ["background",
               "aeroplane",
               "bicycle",
               "bird",
               "boat",
               "bottle",
               "bus",
               "car",
               "cat",
               "chair",
               "cow",
               "diningtable",
               "dog",
               "horse",
               "motorbike",
               "person",
               "pottedplant",
               "sheep",
               "sofa",
               "train",
               "tvmonitor"]

    dataset = torchvision.datasets.VOCDetection(args.data_path, download=False)
    img_np, anns = get_sample(dataset, classes, 4445)
    trans = get_aug([A.Resize(100, 100)])

    labels, bbs = anns
    augmented = trans(**{'image': img_np, 'bboxes': bbs, 'labels': labels})
    img, bbs, labels = augmented['image'], augmented['bboxes'], augmented['labels']
    bb_norm = norm(bbs[0], img.shape[:2])

    model = Model()
    img_tensor = torch.FloatTensor(img / 255.).permute(2,0,1).unsqueeze(0)
    bb_tensor = torch.FloatTensor(bb_norm).unsqueeze(0)

    try:
        logging.info("Load last model weights")
        model = load_model(model, args.wgts_path).to("cpu")
    except FileNotFoundError:
        logging.warning(f"No previous weights in: {args.data_path}, first training cycle")

    logging.info("Start training...")
    fit(model=model,
        X=img_tensor,
        y=bb_tensor,
        epochs=args.epochs,
        lr=args.lr,
        weights_path=args.wgts_path,
        device=DEVICE)
    logging.info("Finished!")