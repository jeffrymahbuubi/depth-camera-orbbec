import albumentations as A
import numpy as np
import os
import glob
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(
        description="Video Augmentation Script"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/video/rgb',
        help="Root directory containing the per-class train folders"
    )
    parser.add_argument(
        '--n_applications',
        type=int,
        default=1,
        help="How many augmented variants to create per original video"
    )
    return parser.parse_args()   # ← you must return the parsed args!

def augment_numpy_video(arr: np.ndarray):
    T, C, H, W = arr.shape
    out_frames, replay = [], None

    transform = A.ReplayCompose([
        A.ElasticTransform(alpha=0.5, p=0.5),
        A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, p=0.5),
        A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(p=0.5),
        A.PixelDropout(drop_value=0, dropout_prob=0.01, p=0.5),
        A.PixelDropout(drop_value=255, dropout_prob=0.01, p=0.5),
        A.Blur(blur_limit=(2, 4), p=0.5)
    ])

    for t in range(T):
        frame = arr[t].transpose(1, 2, 0).astype("uint8")
        if t == 0:
            data   = transform(image=frame)
            new    = data["image"]
            replay = data["replay"]
        else:
            new = A.ReplayCompose.replay(replay, image=frame)["image"]
        out_frames.append(new)

    return np.stack([f.transpose(2, 0, 1) for f in out_frames], axis=0)

def main():
    args = parse_args()  # now args.data_dir and args.n_applications are set

    # Use a separate output folder from the source
    train_dir = os.path.join(args.data_dir, 'train')
    aug_dir   = os.path.join(os.path.dirname(train_dir), 'train')

    # Walk each class folder
    for cls in tqdm(os.listdir(train_dir), desc="Processing classes"):
        src_cls_dir = os.path.join(train_dir, cls)
        dst_cls_dir = os.path.join(aug_dir,   cls)

        # 1) Clear out any previous augmented files
        if os.path.isdir(dst_cls_dir):
            for old in os.listdir(dst_cls_dir):
                if "_aug" in old or old.endswith("_orig.npy"):
                    os.remove(os.path.join(dst_cls_dir, old))
        else:
            os.makedirs(dst_cls_dir, exist_ok=True)

        # 2) Generate fresh augmentations
        file_list = [f for f in os.listdir(src_cls_dir) if f.endswith(".npy")]
        for fname in tqdm(file_list, desc=f"Augmenting {cls}", leave=False):
            src_path = os.path.join(src_cls_dir, fname)
            base, _  = os.path.splitext(fname)

            arr = np.load(src_path)

            # Generate N augmentations
            for i in range(1, args.n_applications + 1):
                aug_arr = augment_numpy_video(arr)
                out_path = os.path.join(dst_cls_dir, f"{base}_aug{i}.npy")
                np.save(out_path, aug_arr)

        print(f"[{cls}] now has {len(os.listdir(dst_cls_dir))} files in {dst_cls_dir}")

    print("✅ Done creating fresh synthetic training videos.")

if __name__ == '__main__':
    main()
