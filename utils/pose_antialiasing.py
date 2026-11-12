"""Exact AA opacity compensation for pose-only passes, with bounded scratch memory.

The pinned Graphdeco backward uses already-filtered covariance entries in the
derivative of det(C) / det(C + 0.3 I), effectively adding 0.3 twice there. Keep
its forward image by applying that factor explicitly and invoking the ordinary
non-AA kernel (which still adds 0.3 I). Only pose passes use this path; existing
Gaussian-training kernels and saved experiments are unchanged.
"""
import math

import torch
from torch.autograd.function import once_differentiable


def compensation(means, scales, rotations, camera):
    """Camera-frame inputs; return sqrt(det(C)/det(C+0.3I)) per Gaussian."""
    # The renderer receives unit quaternions and does not renormalize in CUDA.
    w,x,y,z=rotations.unbind(-1)
    L=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                   2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                   2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)
    L=L*scales[:,None,:]
    # Native rasterization rejects camera-z <= 0.2 before computing covariance.
    # Keep those unused rows finite with zero derivatives.
    depth=means[:,2].clamp_min(.2)
    tx,ty=math.tan(camera.FoVx/2),math.tan(camera.FoVy/2)
    xz=(means[:,0]/depth).clamp(-1.3*tx,1.3*tx)
    yz=(means[:,1]/depth).clamp(-1.3*ty,1.3*ty)
    Jx=(L[:,0]-xz[:,None]*L[:,2])*(camera.image_width/(2*tx)/depth)[:,None]
    Jy=(L[:,1]-yz[:,None]*L[:,2])*(camera.image_height/(2*ty)/depth)[:,None]
    a,b,c=Jx.square().sum(-1),(Jx*Jy).sum(-1),Jy.square().sum(-1)
    factor=((a*c-b*b)/((a+.3)*(c+.3)-b*b).clamp_min(1e-12)).clamp_min(.000025).sqrt()
    return torch.where(means[:,2]>.2,factor,torch.ones_like(factor))[:,None]


class _ChunkedCompensation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,scales,rotations,camera,chunk_size):
        ctx.save_for_backward(means,scales,rotations)
        ctx.camera,ctx.chunk_size=camera,chunk_size
        output=means.new_empty((len(means),1))
        for start in range(0,len(means),chunk_size):
            end=start+chunk_size
            output[start:end]=compensation(means[start:end],scales[start:end],rotations[start:end],camera)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        tensors=ctx.saved_tensors
        needed=ctx.needs_input_grad[:3]
        outputs=[torch.empty_like(value) if need else None for value,need in zip(tensors,needed)]
        # Recompute one chunk's autograd graph at a time rather than keeping
        # projected 3x3 covariance graphs for all ten million LiDAR points.
        for start in range(0,len(tensors[0]),ctx.chunk_size):
            end=start+ctx.chunk_size
            with torch.enable_grad():
                inputs=[value[start:end].detach().requires_grad_(need) for value,need in zip(tensors,needed)]
                value=compensation(*inputs,ctx.camera)
                grads=torch.autograd.grad(value,[p for p,need in zip(inputs,needed) if need],gradient[start:end])
            index=0
            for output in outputs:
                if output is not None:
                    output[start:end]=grads[index]
                    index+=1
        return *outputs,None,None


def pose_compensation(means,scales,rotations,camera,chunk_size=131072):
    if chunk_size<1: raise ValueError('AA compensation chunk size must be positive')
    return _ChunkedCompensation.apply(means,scales,rotations,camera,chunk_size)
