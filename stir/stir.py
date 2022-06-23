'''
A relaxation of inverted_rep_divergence, we calculate CKA on m2(X) and 
m2(X') where X' are inverted reps of m1(X), ie m1(X) \approx m1(X').
'''

from typing import Union, Dict, Optional
import torch
import numpy as np
import os
import argparse, itertools
import output as out
from torchvision.utils import save_image

import model.tools.image_object as io
from attack.losses import SIM_METRICS, LPNormLossSingleModel, SimLoss, CompositeLoss, TVLoss

import helper as hp

def get_seed_images(seed, shape, verbose, inputs=None):
    print (f'From get_seed_images, seed = {seed}')
    if seed == 'super-noise-same':
        if verbose:    print('=> Seeds: Random super-noise, same for all images')
        init_seed_images = torch.randn(*shape[1:], dtype=torch.float)
        init_seed_images = torch.stack([init_seed_images] * shape[0])
    elif seed == 'completely-random':
        if verbose:    print('=> Seeds: Completely random images')
        init_seed_images = torch.randint(0, 255, size=shape[1:], dtype=torch.float)/255.
        init_seed_images = torch.stack([init_seed_images] * shape[0])
    elif seed == 'super-noise':
        if verbose:    print('=> Seeds: Random super-noise')
        init_seed_images = torch.randn(*shape, dtype=torch.float)
    elif seed == 'white':
        if verbose:    print('=> Seeds: All white')
        init_seed_images = torch.ones(*shape, dtype=torch.float)
    elif seed == 'black':
        if verbose:    print('=> Seeds: All black')
        init_seed_images = torch.zeros(*shape, dtype=torch.float)
    elif seed == 'all_types':
        if verbose:    print('=> Seeds: Using all types of seeds')
        all_seeds = ['super-noise-same', 'white', 'black']
        if shape[0] % len(all_seeds) == 0:
            num_samples = [int(shape[0] / len(all_seeds))] * len(all_seeds)
        else:
            num_samples = [int(shape[0] / len(all_seeds))] * len(all_seeds)
            num_samples[-1] += int(shape[0] % len(all_seeds))
        assert sum(num_samples) == shape[0], (num_samples, shape[0])
        init_seed_images = None
        for s, num in zip(all_seeds, num_samples):
            init_seed_images = get_seed_images(s, (num, *shape[1:]), verbose=False) \
                if init_seed_images is None else \
                    torch.cat((init_seed_images, get_seed_images(s, (num, *shape[1:]), verbose=False)))
    elif seed == 'mixed_inputs':
        assert inputs is not None, f'Inputs must be provided for seed = {seed}'
        init_seed_images = torch.flip(inputs, (0,))
    else:
        raise ValueError(f'Seed {seed} not supported!')
    return init_seed_images


def STIR(args, 
         data_path_target: str,
         model1: torch.nn.Module, 
         model2: torch.nn.Module, 
         inputs: Union[tuple, torch.Tensor], 
         devices: list, 
         ve_kwargs: Optional[Dict]=None, 
         norm_type: int=2., 
         seed: str='super-noise', 
         verbose=True, 
         sim_metric: str='linear_CKA', 
         no_opt=False, 
         layer1_num=None, 
         layer2_num=None,
         save_generated_images: bool=True,
         model_type: str=None):
    '''
    Given two models model1 and model2 (both instances of attacker.Attacker),
    and a train_loader, it returns a divergence measure between model1 and model2

    This is done by generating a set of images that are perceived similarly by model1
    and then giving it to model2 and calculating the perception distance there.

    inputs: either a set of images (torch.Tensor) or a tuple of (loader, total_imgs)
    train_loader: loads images on which to calculate divergences
    devices: GPU devices
    seed: starting point for representation inversion
    norm_type: norm used to generate inverted reresentations
    save_generated_images: if inverted images need to be saved to disk
    model_type: used for saving generted images, inferred automatically if None
    '''

    if isinstance(inputs, tuple):
        train_loader, total_imgs = inputs
    else:
        train_loader, total_imgs = \
            [(inputs, torch.arange(len(inputs)))], len(inputs) # to make it an iterable

    if ve_kwargs is None:
        dummy_args = hp.DummyArgs(('lpnorm_type', norm_type, float)) # LpNorm for the representation
        ve_kwargs = {
            'custom_loss': LPNormLossSingleModel(dummy_args),
            'constraint': 'unconstrained',
            'eps': 1000, # put whatever, threat model is unconstrained, so this does not matter
            'step_size': 0.5,
            'iterations': 1000,
            'targeted': True,
            'do_tqdm': verbose,
            'should_normalize': True,
            'use_best': True
        }
    ve_kwargs['layer_num'] = layer1_num

    n_samples_per_iter, img_indices = None, {}
    ## create two objects, one to track OG CKA, one to track STIR
    cka_og, cka_stir = SIM_METRICS[sim_metric](), SIM_METRICS[sim_metric]()
    for index, tup in enumerate(train_loader):
        images, labels = tup
        n_samples_this_iter = len(images)
        if n_samples_per_iter is None:  n_samples_per_iter = len(images)

        raw_indices = torch.arange(len(images))
        for l in set(labels.cpu().numpy()):
            if l not in img_indices:
                img_indices[l] = np.arange(torch.sum(labels == l).item())
            else:
                img_indices[l] = 1 + img_indices[l][-1] + np.arange(torch.sum(labels == l).item())

        (_, images_repr1), _ = model1(images, layer_num=layer1_num, 
            with_latent=True if layer1_num is None else False)
        images_repr1 = images_repr1.detach()
        (_, images_repr2), _ = model2(images, layer_num=layer2_num, 
            with_latent=True if layer2_num is None else False)
        images_repr2 = images_repr2.detach()
        
        cka_og(images_repr1, images_repr2)

        seed_images = get_seed_images(seed, images.shape, verbose, inputs=images).to(
            torch.device(f'cuda:{devices[0]}'))

        (_, seed_reps_1), _ = model1(seed_images, layer_num=layer1_num, 
            with_latent=True if layer1_num is None else False)
        seed_reps_1 = seed_reps_1.detach()
        (_, seed_reps_2), _ = model2(seed_images, layer_num=layer2_num, 
            with_latent=True if layer2_num is None else False)
        seed_reps_2 = seed_reps_2.detach()

        if verbose:
            print ('Initial seed and target rep distance on model1: '
                   f'{torch.mean(torch.norm(images_repr1 - seed_reps_1, p=norm_type, dim=1))}')
            print ('Initial seed and target rep distance on model2: '
                   f'{torch.mean(torch.norm(images_repr2 - seed_reps_2, p=norm_type, dim=1))}')

        if no_opt:
            images_matched = seed_images
        else:
            if model_type is None:
                model_type = f'eps{args.model1_path.split("/eps")[-1].split("/")[0]}' if 'eps' in args.model1_path else 'nonrob'
                if 'trades' in args.model1_path.lower():
                    model_type += 'trades'
                if 'mart' in args.model1_path.lower():
                    model_type += 'mart'
            try:
                rand_seed = args.model1_path.split('_rand_seed_')[1].split('.')[0]
                dataset = args.model1_path.split('checkpoints/')[1].split('/')[0]
            except:
                rand_seed, dataset = None, None
            imgs_path = f'./results/generated_images/{args.target_dataset}/'\
                        f'm1_{args.architecture1}_{dataset}_{model_type}_s{rand_seed}_l{layer1_num}'

            loaded_images, mask = load_tensor_images(imgs_path, img_indices, raw_indices, seed, labels, 
                hp.get_classes_names(args.target_dataset, data_path_target))
            print (f'Loaded {torch.sum(mask)} images from disk ({imgs_path})!')

            if torch.sum(~mask) > 0:
                (_, new_images_matched), _ = model1(
                    inp=seed_images[~mask],
                    target=images_repr1[~mask],
                    make_adv=True,
                    with_image=True,
                    **ve_kwargs) # these images are not normalized
                new_images_matched = new_images_matched.detach()

                img_idx_counts = {l:0 for l in img_indices.keys()}
                for l in labels[mask]:
                    img_idx_counts[l.item()] += 1
                img_indices_for_saving = {}
                for l in img_indices.keys():
                    img_indices_for_saving[l] = img_indices[l][img_idx_counts[l]:]

                if save_generated_images:
                    save_tensor_images(imgs_path, img_indices, seed, new_images_matched, 
                        seed_images[~mask], images[~mask], labels[~mask], 
                        hp.get_classes_names(args.target_dataset, data_path_target))
                
                images_matched = images.clone()
                if loaded_images is not None:
                    images_matched[mask] = loaded_images.to(images.device)
                images_matched[~mask] = new_images_matched
            else:
                images_matched = loaded_images.to(images.device)

        seed_reps_1, seed_reps_2, seed_images, loaded_images, images_repr1, images_repr2 = \
            None, None, None, None, None, None
        torch.cuda.empty_cache()

        _, rep_x = model2(images, with_latent=True if layer2_num is None else False, 
            with_image=False, make_adv=False, layer_num=layer2_num)
        rep_x = rep_x.detach()
        _, rep_y = model2(images_matched, with_latent=True if layer2_num is None else False, 
            with_image=False, make_adv=False, layer_num=layer2_num)
        rep_y = rep_y.detach()
        
        cka_stir(rep_x, rep_y)

        rep_x, rep_y, images, images_matched = None, None, None, None
        torch.cuda.empty_cache()

        if index*n_samples_per_iter + n_samples_this_iter == total_imgs:
            break

    ve_kwargs['custom_loss'].clear_cache()

    return cka_stir.value(), cka_og.value()


def load_tensor_images(path, img_indices, raw_indices, seed_name, labels, classes_name):
    path_result = os.path.join(path, 'result')

    mask = torch.zeros_like(labels).bool()
    loaded_imgs = None
    img_idx_counts = {l:0 for l in img_indices.keys()}
    for label, i in zip(labels, raw_indices):
        idx = img_indices[label.item()][img_idx_counts[label.item()]]
        img_idx_counts[label.item()] += 1

        img_name = f'{int(idx)}_{classes_name[label]}_seed_{seed_name}'
        img_result = f'{path_result}/{img_name}.pkl'
        if os.path.exists(img_result):
            im = io.load_object(img_result).image
            mask[i] = True
            loaded_imgs = im.view(1, *im.shape) if loaded_imgs is None else \
                torch.cat((loaded_imgs, im.view(1, *im.shape)))

    return loaded_imgs, mask


def save_tensor_images(path, img_indices, seed_name, results, seeds, targets, labels, classes_name):
    path = os.path.abspath(path)
    for _d in out.recursive_create_dir(path):
        out.create_dir(_d)
    path_target = os.path.join(path, 'target')
    out.create_dir(path_target)
    path_seed = os.path.join(path, 'seed')
    out.create_dir(path_seed)
    path_result = os.path.join(path, 'result')
    out.create_dir(path_result)

    img_idx_counts = {l:0 for l in img_indices.keys()}
    for result, seed, target, label in zip(results, seeds, targets, labels):
        idx = img_indices[label.item()][img_idx_counts[label.item()]]
        img_idx_counts[label.item()] += 1

        img_name = f'{int(idx)}_{classes_name[label]}_seed_{seed_name}'
        img_target = f'{path_target}/{img_name}.png'
        img_seed = f'{path_seed}/{img_name}.png'
        img_result = f'{path_result}/{img_name}.png'

        save_image(target, img_target)
        save_image(seed, img_seed)
        save_image(result, img_result)

        io.save_object(int(idx), target.cpu(), label.item(), f'{path_target}/{img_name}.pkl')
        io.save_object(int(idx), seed.cpu(), label.item(), f'{path_seed}/{img_name}.pkl')
        io.save_object(int(idx), result.cpu(), label.item(), f'{path_result}/{img_name}.pkl')

    print(f'=> Saved images in {path}')

