import argparse
from dutils import store_joint_poses

def parse_args():
    parser = argparse.ArgumentParser(description="Create joint pose files from smpl params")
    parser.add_argument("src", type=str, help="Path to the AMASS dataset folder")
    parser.add_argument("dst", type=str, help="Path to the body joint pose destination folder")
    parser.add_argument("--smpl", type=str, help="Path to the smpl file")
    parser.add_argument("--sk_type", type=str, choices=['nturgbd', 'h36m'], default='nturgbd', help="type of the destination skeleton format")
    parser.add_argument("--only_existing", action="store_true", default=False, help="Create only existing files in the dataset")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    store_joint_poses(args.smpl, args.dst, args.src, args.sk_type, args.only_existing)

