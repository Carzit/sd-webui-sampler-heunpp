import sys
import functools
import torch
from torch import nn
from tqdm import trange

import k_diffusion.sampling # type: ignore
from modules import shared
from modules import sd_samplers, sd_samplers_common
import modules.sd_samplers_kdiffusion as K

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
def sample_heunpp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        elif sigmas[i + 2] == 0:

            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2


            x = x + d_prime * dt

        else:
            # Heun++
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            dt_2 = sigmas[i + 2] - sigmas[i + 1]


            x_3 = x_2 + d_2 * dt_2
            denoised_3 = model(x_3, sigmas[i + 2] * s_in, **extra_args)
            d_3 = to_d(x_3, sigmas[i + 2], denoised_3)

            d_prime = (d + d_2 + d_3) / 3
            x = x + d_prime * dt


    return x

class KDiffusionSamplerLocal(K.KDiffusionSampler):
    
    def __init__(
        self,
        funcname: str,
        original_funcname: str,
        func,
        sd_model: nn.Module
    ):
        # here we do not call super().__init__() 
        # because target function is not in k_diffusion.sampling
        
        denoiser = k_diffusion.external.CompVisVDenoiser if sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser

        self.model_wrap = denoiser(sd_model, quantize=shared.opts.enable_quantization)
        self.funcname = funcname
        self.func = func
        self.extra_params = K.sampler_extra_params.get(original_funcname, [])
        self.model_wrap_cfg = K.CFGDenoiser(self.model_wrap)
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config = None
        self.last_latent = None

        self.conditioning_key = sd_model.model.conditioning_key # type: ignore


def add_heunpp_test():
    original = [ x for x in K.samplers_k_diffusion if x[0] == 'Heun' ][0]
    o_label, o_constructor, o_aliases, o_options = original
    
    label = 'Heun++test0'
    funcname = sample_heunpp.__name__
    
    def constructor(model: nn.Module):
        return KDiffusionSamplerLocal(funcname, o_constructor, sample_heunpp, model)
    
    aliases = [ x + '++test0' for x in o_aliases ]
    
    options = { **o_options }
    
    data = sd_samplers_common.SamplerData(label, constructor, aliases, options)
    
    if len([ x for x in sd_samplers.all_samplers if x.name == label ]) == 0:
        sd_samplers.all_samplers.append(data)


def update_samplers():
    sd_samplers.set_samplers()
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}


def hook(fn):
    
    @functools.wraps(fn)
    def f(*args, **kwargs):
        old_samplers, mode, *rest = args
        
        if mode not in ['txt2img', 'img2img']:
            print(f'unknown mode: {mode}', file=sys.stderr)
            return fn(*args, **kwargs)
        
        update_samplers()
        
        new_samplers = (
            sd_samplers.samplers if mode == 'txt2img' else
            sd_samplers.samplers_for_img2img
        )
        
        return fn(new_samplers, mode, *rest, **kwargs)
    
    return f


# register new sampler
add_heunpp_test()
update_samplers()


# hook Sampler textbox creation
from modules import ui

ui.create_sampler_and_steps_selection = hook(ui.create_sampler_and_steps_selection)
