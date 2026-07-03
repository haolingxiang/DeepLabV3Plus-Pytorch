from tqdm import tqdm
import network
import utils
import os
import argparse
from pathlib import Path

from datasets import VOCSegmentation, Cityscapes, CombinedLandCover
from torchvision import transforms as T
from utils.tiled_inference import infer_image
from utils.inference_export import export_prediction, load_combined_class_info

import torch
import torch.nn as nn

from PIL import Image
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'combined'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--tile_size", type=int, default=1024,
                        help="tile size for pad/slide-window inference (default: 1024)")
    parser.add_argument("--stride", type=int, default=1024,
                        help="sliding window stride for large images (default: 1024)")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="combine_data root for loading class_names.json (combined dataset)",
    )
    parser.add_argument(
        "--no_save_pixel_zh",
        action="store_true",
        default=False,
        help="skip per-pixel Chinese class npz (saves disk space)",
    )

    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    class_info = None
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'combined':
        opts.num_classes = 8
        decode_fn = CombinedLandCover.decode_target
        class_info = load_combined_class_info(opts.data_root)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            mode = "slide" if (h > opts.tile_size or w > opts.tile_size) else "pad"
            pred = infer_image(
                model,
                img,
                transform,
                device,
                num_classes=opts.num_classes,
                tile_size=opts.tile_size,
                stride=opts.stride,
            )
            if opts.save_val_results_to:
                export_prediction(
                    pred,
                    img_name,
                    Path(opts.save_val_results_to),
                    decode_fn,
                    class_info=class_info,
                    save_pixel_zh=not opts.no_save_pixel_zh,
                )
            tqdm.write(f"{img_name}: {w}x{h} -> {mode}, pred {pred.shape[1]}x{pred.shape[0]}")

if __name__ == '__main__':
    main()
