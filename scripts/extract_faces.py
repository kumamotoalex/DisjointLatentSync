import argparse
import torch
from latentsync.utils.util import read_video
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
import tqdm


def extract_faces(video_path: str, output_path: str, height: int = 256):
    """Extract faces from video and save affine transform data.

    Args:
        video_path: Path to input video
        output_path: Path to save affine transform data
        height: Height for face processing (default: 256)
    """
    # Initialize image processor
    mask_image = load_fixed_mask(height)
    image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)

    # Read video frames
    print(f"Reading video from {video_path}")
    video_frames = read_video(video_path, use_decord=False)

    # Extract faces
    faces = []
    boxes = []
    affine_matrices = []
    print(f"Affine transforming {len(video_frames)} faces...")

    for frame in tqdm.tqdm(video_frames):
        face, box, affine_matrix = image_processor.affine_transform(frame)
        faces.append(face)
        boxes.append(box)
        affine_matrices.append(affine_matrix)

    faces = torch.stack(faces)

    # Save affine transform data
    torch.save(
        {"faces": faces, "boxes": boxes, "affine_matrices": affine_matrices},
        output_path,
    )
    print(f"Saved affine transform data to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save affine transform data",
    )
    parser.add_argument(
        "--height", type=int, default=256, help="Height for face processing"
    )
    args = parser.parse_args()

    extract_faces(args.video_path, args.output_path, args.height)


if __name__ == "__main__":
    main()
