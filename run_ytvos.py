import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on Ref-YouTube-VOS dataset")
    parser.add_argument("--mode", type=str, default="mask_crop", choices=["mask_crop", "object_box_crop", "full_frame"],
                        help="CLIP Reranker mode")
    parser.add_argument("--w_finder", type=float, default=0.5, help="Finder score weight (w1)")
    parser.add_argument("--w_clip", type=float, default=0.5, help="CLIP score weight (w2)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--dataset_path", type=str, default="../DB/RVOS/YTVOS", help="Path to Ref-YouTube-VOS dataset")
    return parser.parse_args()


def test(args):

    # initialize EVF-SAM
    tokenizer, evfsam = init_models()

    # initialize Alpha-CLIP
    clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth='weights/clip_l14_336_grit_20m_4xe.pth', device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # initialize Cutie
    cutie = get_default_model(config='ytvos_config')
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    # load videos
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir, 'Ref_YTVOS_val')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    root = args.dataset_path
    img_folder = os.path.join(root, 'valid', 'JPEGImages')
    meta_file = os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json')
    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_test_videos = set(data.keys())
    test_meta_file = os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json')
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])

    # inference
    for idx_, video in enumerate(video_list):
        metas = []
        expressions = data[video]['expressions']
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        for i in range(num_expressions):
            meta = {}
            meta['video'] = video
            meta['exp'] = expressions[expression_list[i]]['exp']
            meta['exp_id'] = expression_list[i]
            meta['frames'] = data[video]['frames']
            metas.append(meta)
        meta = metas
        video_name = video
        frames = data[video]['frames']
        video_len = len(frames)

        # CHECK IF ALREADY EVALUATED (Resume capability)
        already_done = True
        for e in range(num_expressions):
            exp_id = meta[e]['exp_id']
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                already_done = False
                break
            for frame in frames:
                if not os.path.exists(os.path.join(save_path, frame + '.png')):
                    already_done = False
                    break
            if not already_done:
                break
        
        if already_done:
            print(f"Video {idx_+1}/{len(video_list)}: {video} - Already evaluated. Skipping.")
            continue

        print(f"Video {idx_+1}/{len(video_list)}: {video}")

        # input pre-process
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        for i in range(video_len):
            img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            # BEiT pre-process
            img_beit = beit3_preprocess(Image.open(img_path), 224)
            imgs_beit.append(img_beit)

            # SAM pre-process
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            # Alpha-CLIP pre-process
            img_clip = clip_preprocess(Image.open(img_path))
            imgs_clip.append(img_clip)

            # Cutie pre-process
            img_cutie = tv.transforms.ToTensor()(Image.open(img_path))
            imgs_cutie.append(img_cutie)

        # for each language
        for e in range(num_expressions):

            # make files
            video_name = meta[e]['video']
            exp = meta[e]['exp']
            exp_id = meta[e]['exp_id']
            frames = meta[e]['frames']
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # per-frame mask prediction
            ref_masks = []
            ref_scores = []
            ref_num = 5
            for ref_idx in range(ref_num):
                i = int(ref_idx * (video_len - 1) / (ref_num - 1))
                words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
                ref_mask, ref_score = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
                ref_mask = (ref_mask > 0).float()
                ref_masks.append(ref_mask)

                # consider vision-text alignment in addition to segmentation confidence
                w1, w2 = args.w_finder, args.w_clip
                clip_text = alphaclip.tokenize([exp]).cuda()
                ref_img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
                image_np = cv2.imread(ref_img_path)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

                clip_sim = compute_clip_similarity(
                    clip, clip_preprocess, clip_preprocess_mask,
                    image_np, ref_mask, clip_text, mode=args.mode
                )
                ref_score = w1 * ref_score + w2 * clip_sim
                ref_scores.append(ref_score)

            # select reference frame with highest mask score
            best_ref_idx = torch.argmax(torch.stack(ref_scores, dim=0), dim=0)
            best_i = int(best_ref_idx * (video_len - 1) / (ref_num - 1))

            # forward pass
            for i in range(best_i, video_len):
                if i == best_i:
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == video_len - 1:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)

            # backward pass
            for i in range(best_i, -1, -1):
                if i == best_i:
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == 0:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        test(args)
