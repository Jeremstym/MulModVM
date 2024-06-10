# This file is a modified version of the original dataset_tool.py file from the StyleGAN2-ADA-PyTorch repository.

import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import torchvision.transforms as transforms
import torch

import click
import numpy as np
import PIL.Image
from tqdm import tqdm

def grab_image_augmentations(img_size: int, target: str, crop_scale_lower: float = 0.08) -> transforms.Compose:
  """
  Defines augmentations to be used with images during contrastive training and creates Compose.
  """
  if target.lower() == 'dvm':
    transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
      transforms.RandomResizedCrop(size=(img_size,img_size), scale=(crop_scale_lower, 1.0), ratio=(0.75, 1.3333333333333333)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ToTensor(),
      transforms.Lambda(lambda x : x.float())
    ])
  else:
    raise ValueError(f'Unknown target {target}')
  return transform

def grab_default_transform(img_size: int) -> transforms.Compose:
    """
    Defines default transformations for images.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(img_size,img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x : x.float())
        ])
    return transform

def grad_image_eval_transform(img_size: int, target: str) -> transforms.Compose:
    if target.lower() == 'dvm':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
            transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(size=(img_size,img_size), antialias=False),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.float())
        ])
    else:
        raise ValueError(f'Unknown target {target}')
    return transform

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int], label_path: Optional[str] = None):
    if os.path.isdir(source):
        if source.rstrip('/').endswith('_lmdb'):
            return open_lmdb(source, max_images=max_images)
        else:
            return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if os.path.basename(source) == 'cifar-10-python.tar.gz':
            return open_cifar10(source, max_images=max_images)
        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':
            return open_mnist(source, max_images=max_images)
        elif file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        elif file_ext(source) == 'pt':
            return open_pickle(source, max_images=max_images, label_path=label_path)
        else:
            assert False, 'unknown archive type'
    else:
        error(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pylint: disable=import-error
    import lmdb  # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx-1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fname))
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

def open_pickle(source_file, *, max_images: Optional[int], label_path: Optional[str] = None):
    # with open(source_file, 'rb') as file:
    #     data = pickle.load(file)

    data = torch.load(source_file)
    if label_path is not None:
        print(f'Loading labels from {label_path}')
        labels = {}
        with open(label_path, 'r') as file:
            labels = json.load(file)["labels"]
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                raise ValueError('Labels file is empty') # exceptionnaly raise an error here
                labels = {}

    max_idx = maybe_min(len(data), max_images)

    def iterate_images():
        for idx, img_name in enumerate(data):
            img = np.array(PIL.Image.open(img_name))
            yield dict(img=img, label=labels.get(img_name))
            if labels.get(img_name) is None:
                print(f'No label found for {img_name}')
                raise ValueError('No label found for image')
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()


#----------------------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--labels', help='Directory or archive name for input labels', type=str, default=None, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', metavar='WxH', type=parse_tuple)
def convert_dataset(
    ctx: click.Context,
    source: str,
    labels: Optional[str],
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
):

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images, label_path=labels)
    print(f'Converting {num_files} images')
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    print(f'Saving to {archive_root_dir}')

    if resolution is None: resolution = (None, None)
    # transform_image = make_transform(transform, *resolution)
    transform_image = grab_image_augmentations(resolution[0], 'dvm')
    resize_image = grab_default_transform(resolution[0])

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files, desc='Converting images'):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        img = image['img']
        # Apply crop and resize.
        img = transform_image(img)
        print(f"img shape: {img.shape}")
        print(f"img type: {type(img)}")
        print(f"img max value: {img.max()}")
        raise Exception
        unaugmented_image = resize_image(img)

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        # check if img is tensor type
        if isinstance(img, torch.Tensor):
            channels = img.shape[0]
        else:
            channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1] if isinstance(img, np.ndarray) else img.size(2),
            'height': img.shape[0] if isinstance(img, np.ndarray) else img.size(1),
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                error('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()] # pylint: disable=unsubscriptable-object
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
            image_bits = io.BytesIO()
            img.save(image_bits, format='png', compress_level=0, optimize=False)
            save_bytes(os.path.join(archive_root_dir, "augmented", archive_fname), image_bits.getbuffer())
            if image['label'] is None:
                print(f'No label found for {archive_fname}')
                raise ValueError('No label found for image')
            labels.append([archive_fname, image['label']] if image['label'] is not None else None)
        elif isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
            image_bits = io.BytesIO()
            img.save(image_bits, format='png', compress_level=0, optimize=False)
            save_bytes(os.path.join(archive_root_dir, "augmented", archive_fname), image_bits.getbuffer())
            labels.append([archive_fname, image['label']] if image['label'] is not None else None)
            if image['label'] is None:
                print(f'No label found for {archive_fname}')
                raise ValueError('No label found for image')
        else:
            raise ValueError('img must be either a numpy array or a torch tensor')

        # Save the unaugmented image as an uncompressed PNG.
        if isinstance(unaugmented_image, np.ndarray):
            unaugmented_image = PIL.Image.fromarray(unaugmented_image, { 1: 'L', 3: 'RGB' }[channels])
            image_bits = io.BytesIO()
            unaugmented_image.save(image_bits, format='png', compress_level=0, optimize=False)
            save_bytes(os.path.join(archive_root_dir, "unaugmented", f'{idx_str[:5]}/img{idx_str}_unaugmented.png'), image_bits.getbuffer())
        elif isinstance(unaugmented_image, torch.Tensor):
            unaugmented_image = transforms.ToPILImage()(unaugmented_image)
            image_bits = io.BytesIO()
            unaugmented_image.save(image_bits, format='png', compress_level=0, optimize=False)
            save_bytes(os.path.join(archive_root_dir, "unaugmented", f'{idx_str[:5]}/img{idx_str}_unaugmented.png'), image_bits.getbuffer())
        else:
            raise ValueError('img must be either a numpy array or a torch tensor')

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
