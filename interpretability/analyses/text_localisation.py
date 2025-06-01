import os
import argparse
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from bcos.experiments.utils import Experiment
from bcos.common import get_inx2label_imagenet
import bcos.data.transforms as custom_transforms
import bcos.data.imagenet as imagenet 
from PIL import Image
import torchvision.transforms as transforms


def load_model(exp_name, use_attn_unpool):
    exp = Experiment(exp_name)
    model = exp.load_trained_model()
    if use_attn_unpool:
        model.model.attnpool.attn_unpool = True
    return model


def prepare_data(exp_name_data):
    exp_data = Experiment(exp_name_data)
    dm = exp_data.get_datamodule()
    dm.setup('val')
    return dm.val_dataloader()


def get_clip_model():
    clip_model, preprocess = clip.load("RN50")
    clip_model.float()
    clip_model.eval()
    return clip_model


def select_random_images(loader, max_imgs=32, seed=42):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(high=len(loader.dataset), size=(max_imgs,), generator=g)
    idx = idx[:max_imgs]

    imgs, lbls = [], []
    for i in idx:
        img, lbl = loader.dataset[i]
        imgs.append(img)
        lbls.append(lbl)
    imgs = torch.stack(imgs)
    lbls = torch.tensor(lbls)
    return imgs, lbls


def transform_images(imgs):
    transform_bcos = custom_transforms.AddInverse()
    return transform_bcos(imgs)


def tokenize_text(clip_model, templates, test_text_name):
    test_text = [template.format(test_text_name) for template in templates]
    test_text = clip.tokenize(test_text).cuda()
    class_embeddings = clip_model.encode_text(test_text)
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding = class_embeddings.mean(dim=0)
    class_embedding /= class_embedding.norm()
    return class_embedding.unsqueeze(1)


def compute_attributions(model, test_img, zeroshot_weight, smooth=0, alpha_percentile=99.5, pool_cosine=1, norm_max_cosine=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.enable_grad(), model.explanation_mode(), torch.autograd.set_detect_anomaly(True):
        imga = test_img[None].to(device).requires_grad_()
        outa = model(imga)

        img_features = outa / outa.norm(dim=-1, keepdim=True)
        logits = img_features @ zeroshot_weight

        if model.model.attnpool.attn_unpool:
            logits = logits.reshape(-1, 1)
            if pool_cosine == 0:
                num_features = logits.shape[0]
                logits = logits.reshape(-1, num_features)
                max_locations = logits.argmax(dim=1)
                mask = torch.zeros_like(logits)
                for i in range(logits.shape[0]):
                    mask[i, max_locations[i]] = 1.0
                logits = logits * mask.detach()
                logits = logits.reshape(1, num_features)
            if norm_max_cosine:
                logits = logits / logits.abs().detach().max(dim=0, keepdim=True)[0]
            if pool_cosine > 1:
                logits = logits * torch.pow(logits, pool_cosine-1).abs().detach()
            logits = logits.mean(dim=0)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        logits.max(1).values.backward(inputs=[imga])

        grada = imga.grad
        imga = imga.detach().cpu()[0]
        grada = grada.detach().cpu()[0]

    contribs = (imga * grada).sum(0, keepdim=True)

    rgb_grad = grada / (grada.abs().max(0, keepdim=True).values + 1e-12)
    rgb_grad = rgb_grad.clamp(min=0)
    rgb_grad = rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12)

    alpha = grada.norm(p=2, dim=0, keepdim=True)
    alpha = torch.where(contribs < 0, 1e-12, alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
    alpha = (alpha / torch.quantile(alpha, q=alpha_percentile / 100)).clip(0, 1)

    rgb_grad = torch.cat([rgb_grad, alpha], dim=0)
    grad_image = rgb_grad.permute(1, 2, 0).detach().cpu().numpy()

    contribs = contribs.detach().cpu().numpy().squeeze()
    cutoff = np.percentile(np.abs(contribs), 99.5)
    contribs = np.clip(contribs, -cutoff, cutoff)
    vrange = np.max(np.abs(contribs.flatten()))

    return grad_image, contribs, vrange


def save_visualizations(fig, axs, row, original_img, class_name, grad_image, contribs, vrange, caption, use_class_name=False):
    cmap = plt.cm.bwr

    if isinstance(original_img, Image.Image):
        original_img = transforms.ToTensor()(original_img)

    if use_class_name:
        if row == 0:
            axs[0].imshow(original_img.permute(1, 2, 0).cpu().numpy())
            axs[0].axis('off')
            axs[0].set_title(f"Original: {class_name}", fontsize=12)
        else:
            axs[0].axis('off')

        axs[1].imshow(grad_image)
        axs[1].axis('off')
        axs[1].set_title(f"B-cos explanations: {caption}", fontsize=12)

        im = axs[2].imshow(contribs, cmap=cmap, vmin=-vrange, vmax=vrange)
        axs[2].axis('off')
        axs[2].set_title(f"Raw Attributions: {caption}", fontsize=12)
    else:
        if row == 0:
            axs[row, 0].imshow(original_img.permute(1, 2, 0).cpu().numpy())
            axs[row, 0].axis('off')
            axs[row, 0].set_title(f"Original: {class_name}", fontsize=12)
        else:
            axs[row, 0].axis('off')

        axs[row, 1].imshow(grad_image)
        axs[row, 1].axis('off')
        axs[row, 1].set_title(f"B-cos explanations: {caption}", fontsize=12)

        im = axs[row, 2].imshow(contribs, cmap=cmap, vmin=-vrange, vmax=vrange)
        axs[row, 2].axis('off')
        axs[row, 2].set_title(f"Raw Attributions: {caption}", fontsize=12)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--text_to_localize", type=str, required=False)
    parser.add_argument("--exp_name_data", type=str, 
                        default='experiments/ImageNet/clip_bcosification/resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification')
    parser.add_argument("--image_index", type=int, default=11)
    parser.add_argument("--use_attn_unpool", action="store_true", default=False)
    parser.add_argument("--pool_cosine", type=int, default=1)
    parser.add_argument("--norm_max_cosine", action="store_true", default=False)
    parser.add_argument("--smooth", type=int, default=0)
    parser.add_argument("--random_img_path", type=str, default=None)
    parser.add_argument("--use_class_name", action="store_true", default=False)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    model = load_model(args.exp_name, args.use_attn_unpool)
    clip_model = get_clip_model()

    if args.random_img_path is not None:
        image = Image.open(args.random_img_path).convert('RGB')
        original_img = image.copy()
        def _convert_image_to_rgb(image):
            return image.convert("RGB")
        transform = transforms.Compose([
                        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        _convert_image_to_rgb,
                        transforms.ToTensor(),
                        custom_transforms.AddInverse(),
                    ])
        image = transform(image)
        test_img = image
        class_name = os.path.splitext(os.path.basename(args.random_img_path))[0]
    else:
        loader = prepare_data(args.exp_name_data)
        imgs, lbls = select_random_images(loader)
        original_imgs = imgs.clone()
        imgs = transform_images(imgs)

        img_indx = args.image_index
        test_img = imgs[img_indx]
        original_img = original_imgs[img_indx]
        lbl_indx = lbls[img_indx]
        class_name = get_inx2label_imagenet(lbl_indx.item())

    if args.use_class_name:
        text_to_localize_list = [class_name]
    else:
        text_to_localize_list = args.text_to_localize.split(',')

    fig, axs = plt.subplots(len(text_to_localize_list), 3, figsize=(15, 5 * len(text_to_localize_list)))
    fig.suptitle("B-cosified CLIP text-based localisation", fontsize=16, fontweight='bold')

    for i, test_text_name in enumerate(text_to_localize_list):
        zeroshot_weight = tokenize_text(clip_model, imagenet.imagenet_templates, test_text_name)
        grad_image, contribs, vrange = compute_attributions(model, test_img, zeroshot_weight, args.smooth, 
                                                            pool_cosine=args.pool_cosine, norm_max_cosine=args.norm_max_cosine)
        if args.use_class_name or len(text_to_localize_list) == 1:
            im = save_visualizations(fig, axs, i, original_img, class_name, grad_image, contribs, vrange, test_text_name,True)
        else:
            im = save_visualizations(fig, axs, i, original_img, class_name, grad_image, contribs, vrange, test_text_name, False)

    if args.save_path is None:
        args.save_path = args.exp_name

    if args.random_img_path is not None:
        final_plots_dir = f"{args.save_path}/textloc_expl/random_image/{args.exp_name.split('/')[-1]}/smooth{args.smooth}"
    else:
        final_plots_dir = f"{args.save_path}/textloc_expl/{class_name}_{str(lbl_indx.item())}_{str(img_indx)}/{args.exp_name.split('/')[-1]}/smooth{args.smooth}"

    if args.use_attn_unpool:
        final_plots_dir += f"_attn_unpool"
    if args.pool_cosine != 1:
        final_plots_dir += f"_pool_cosine{args.pool_cosine}"
    if args.norm_max_cosine:
        final_plots_dir += f"_norm_max_cosine"

    os.makedirs(final_plots_dir, exist_ok=True)

    if args.random_img_path is not None:
        plt.savefig(os.path.join(final_plots_dir, f"File:{class_name}_Text:{args.text_to_localize}.pdf"), dpi=200, bbox_inches='tight')
    else:
        if args.use_class_name:
            plt.savefig(os.path.join(final_plots_dir, f"Text:{str(lbl_indx.item())}_Text:{class_name}.pdf"), dpi=200, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(final_plots_dir, f"Text:{str(lbl_indx.item())}_Text:{args.text_to_localize}.pdf"), dpi=200, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
