import numpy as np
import torch

def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    # vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1/np.sqrt(s)))
    return W, -mu

def normalize(vecs):
    """标准化
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    
    return vecs, normalize(vecs)


def apply_whitening(vecs, n_components):
    kernel, bias = compute_kernel_bias(vecs)
    print(kernel.dtype, bias.dtype)
    kernel = kernel[:, :n_components]
    params = {}
    params['whiten'] = (kernel, bias)
    

    vecs, embeddings = transform_and_normalize(vecs, 
                kernel=params['whiten'][0],
                bias=params['whiten'][1]
            )  # whitening
    
    return params, vecs, embeddings
  
  
vecs = torch.randn(256, 768)
vecs = vecs / vecs.norm(p=2, dim=-1, keepdim=True)
n_components = 64
params, vecs, whiten_embeddings = apply_whitening(vecs.numpy(), n_components)
