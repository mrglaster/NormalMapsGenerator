import numpy as np
import onnx
import os
import infer
import onnxruntime as ort
import onnxoptimizer
import argparse
import sys
from PIL import Image
onnx_model = onnx.load("./model/deepbump256.onnx")
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
optimized_model = onnxoptimizer.optimize(onnx_model, passes)
onnx.save(optimized_model, "./model/deepbump256.onnx")

def make_normalmap(img_src, bump_path, overlap):
    #based on https://github.com/HugoTini/DeepBump
    img = np.array(Image.open(img_src)) / 255.0
    img = np.transpose(img, [2, 0, 1])
    img = np.mean(img[0:3], axis=0, keepdims=True)
    print('Tilling of texture: ', img_src)
    tile_size = 256
    overlaps = {'small': tile_size // 6, 'medium': tile_size // 4, 'large': tile_size // 2}
    stride_size = tile_size - overlaps[overlap]
    tiles, paddings = infer.tiles_split(img, (tile_size, tile_size),(stride_size, stride_size))
    print('Generating normal map for texture: ', img_src)
    ort_session = ort.InferenceSession('./model/deepbump256.onnx')
    pred_tiles = infer.tiles_infer(tiles, ort_session)
    print('Merging tiles of texture: ', img_src)
    pred_img = infer.tiles_merge(pred_tiles, (stride_size, stride_size), (3, img.shape[1], img.shape[2]), paddings)
    pred_img = pred_img.transpose((1, 2, 0))
    pred_img = Image.fromarray((pred_img * 255.0).astype(np.uint8))
    print("Result was saved to: "+bump_path)
    pred_img.save(bump_path)


def is_imagefile(i):
    i = i.lower()
    if i.endswith('.bmp') or i.endswith('.tga') or i.endswith('.png') or i.endswith(".jpeg") or i.endswith(".jpg"):
        return True
    else:
        return False


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-olp', '--overlap', type=str, required=True)
    return parser


def main():
    parser = argparser()
    namespace = parser.parse_args(sys.argv[1:])
    input_data =format(namespace.input)
    overlap = format(namespace.overlap)
    overlaps = ['small', 'medium', 'large']
    if overlap in overlaps:
        if os.path.exists(input_data) and os.path.isdir(input_data):
            for i in os.listdir(input_data):
                if is_imagefile(i):
                    try:
                        make_normalmap(img_src=i,bump_path=i[:len(i)-4]+'_normal.png', overlap=overlap)
                    except:
                        print("Something went wrong with: "+i)
                        return
        elif os.path.exists(input_data):
            print("Path exists!")
            try:
                print("test")
                make_normalmap(img_src=input_data, bump_path=input_data[:len(input_data) - 4] + '_normal.png', overlap=overlap)
            except:
                print("Something went wrong with: "+input_data)
                return
    else:
        print("Not allowed overlap")
        print('Possible overlap: "small", "medium", "large" ')
        return



if __name__ == '__main__':
    main()
