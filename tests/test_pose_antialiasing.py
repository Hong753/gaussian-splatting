"""Exact AA adjoints and a differentiable pose probe independent of CUDA."""
import math
from types import SimpleNamespace
import unittest

import torch

from utils.pose_antialiasing import compensation,pose_compensation
from utils.pose_refinement import rotation_quaternion,rotation_matrix,corrected_transform,quaternion_product,base_quaternion
from utils.graphics_utils import getProjectionMatrix
from scripts.pose_gradient_check import probe_model,single_splat_patch


def camera():
    R=rotation_matrix(rotation_quaternion(torch.tensor([.12,-.08,.05])))
    center=torch.tensor([.1,-.1,0.])
    w2c=torch.eye(4);w2c[:3,:3]=R;w2c[:3,3]=-R@center
    return SimpleNamespace(FoVx=math.pi/2,FoVy=math.pi/2,image_width=48,image_height=48,
        camera_center=center,world_view_transform=w2c.T.contiguous(),
        projection_matrix=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).T.contiguous())


class PoseAntialiasingTests(unittest.TestCase):
    def test_chunked_compensation_gradcheck_and_matrix_covariance_reference(self):
        cam=camera()
        means=torch.tensor([[.1,-.2,2.],[-.15,.25,2.4]],dtype=torch.float64,requires_grad=True)
        scales=torch.tensor([[.1,.07,.04],[.2,.04,.09]],dtype=torch.float64,requires_grad=True)
        rotations=rotation_quaternion(torch.tensor([[.2,-.1,.3],[-.4,.2,.1]],dtype=torch.float64)).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda a,b,c:pose_compensation(a,b,c,cam,chunk_size=1),
                                                (means,scales,rotations),eps=1e-6,atol=1e-5,rtol=1e-4))
        # Independent matrix multiplication, instead of production row products.
        R=rotation_matrix(rotations)
        cov=R@torch.diag_embed(scales.square())@R.transpose(-1,-2)
        z=means[:,2];zero=torch.zeros_like(z)
        J=torch.stack((24/z,zero,-24*means[:,0]/z.square(),zero,24/z,-24*means[:,1]/z.square()),-1).reshape(-1,2,3)
        C=J@cov@J.transpose(-1,-2)
        expected=(torch.linalg.det(C)/torch.linalg.det(C+.3*torch.eye(2,dtype=C.dtype))).sqrt()[:,None]
        torch.testing.assert_close(pose_compensation(means,scales,rotations,cam,1),expected,atol=1e-12,rtol=1e-12)
        for chunk in (1,3):
            result=pose_compensation(means,scales,rotations,cam,chunk)
            direct=compensation(means,scales,rotations,cam)
            a=torch.autograd.grad(result.sum(),(means,scales,rotations),retain_graph=True)
            b=torch.autograd.grad(direct.sum(),(means,scales,rotations),retain_graph=True)
            for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)

    def test_frozen_scales_and_near_culled_rows_have_finite_gradients(self):
        cam=camera()
        means=torch.tensor([[0.,0.,0.],[.1,.1,-1.],[.1,.1,2.]],dtype=torch.float64,requires_grad=True)
        scales=torch.full((3,3),.1,dtype=torch.float64)
        rotations=torch.tensor([[1.,0,0,0]]*3,dtype=torch.float64,requires_grad=True)
        value=pose_compensation(means,scales,rotations,cam,2)
        grads=torch.autograd.grad(value.sum(),(means,rotations))
        self.assertTrue(torch.equal(value[:2],torch.ones_like(value[:2])))
        for gradient in grads:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient[:2]==0).all())

    def test_pinned_native_aa_formula_double_adds_filter_variance(self):
        values=torch.tensor([.8,.2,1.4],dtype=torch.float64,requires_grad=True)
        a,b,c=values;h=.3
        factor=((a*c-b*b)/((a+h)*(c+h)-b*b)).sqrt()
        correct=torch.autograd.grad(factor,values)[0]
        x,y,z=a.detach()+h,c.detach()+h,b.detach()
        coefficient=(.5/factor.detach())/(h*h+h*(x+y)+x*y-z*z).square()
        wrong=torch.stack((h*(h*y+y*y+z*z),-2*h*z*(h+x+y),h*(h*x+x*x+z*z)))*coefficient
        self.assertGreater(float(((wrong-correct).abs()/correct.abs()).max()),.1)
        # Actual finite differences agree with the pre-filter derivative.
        def f(v):
            a,b,c=v
            return ((a*c-b*b)/((a+h)*(c+h)-b*b)).sqrt()
        numeric=[]
        for axis in range(3):
            offset=torch.zeros_like(values);offset[axis]=1e-5
            numeric.append((f(values.detach()+offset)-f(values.detach()-offset))/2e-5)
        torch.testing.assert_close(correct,torch.stack(numeric),atol=1e-9,rtol=1e-7)

    def test_single_splat_probe_all_six_derivatives_and_cutoff_margins(self):
        cam=camera();model=probe_model('cpu');bg=torch.zeros(3)
        for aa in (False,True):
            t=torch.tensor([.005,-.003,.002],dtype=torch.float64,requires_grad=True)
            r=torch.tensor([.001,-.002,.0015],dtype=torch.float64,requires_grad=True)
            result=single_splat_patch(cam,model,bg,t,r,aa)
            weights=torch.linspace(.2,1.5,result.numel(),dtype=torch.float64).reshape_as(result)
            analytic=torch.cat(torch.autograd.grad((result*weights).mean(),(t,r)))
            self.assertTrue((analytic.abs()>1e-6).all())
            for epsilon in (.001,.0005):
                numeric=[]
                for axis in range(6):
                    values=[]
                    for sign in (-1,1):
                        a,b=t.detach().clone(),r.detach().clone()
                        (a if axis<3 else b)[axis%3]+=sign*epsilon
                        values.append((single_splat_patch(cam,model,bg,a,b,aa)*weights).mean())
                    numeric.append((values[1]-values[0])/(2*epsilon))
                torch.testing.assert_close(analytic,torch.stack(numeric),atol=2e-5,rtol=.05)


if __name__=='__main__': unittest.main()
