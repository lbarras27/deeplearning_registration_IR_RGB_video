import argparse
import util

parser = argparse.ArgumentParser()

parser.add_argument("--path_data", "-p", help="Path to the directory containing all the videos")

parser.add_argument("--dest_path", "-d", help="name of the directory we want to have the results")

args = parser.parse_args()

if args.path_data:
    path_DATA = args.path_data

if args.dest_path:
    path_dest = args.dest_path
    
util.convert_each_videos_in_frames(path_DATA, path_dest)
    