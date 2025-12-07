"""Some standard gradient-based stochastic optimizers.

These are just standard routines that don't make any use of autograd,
though you could take gradients of these functions too if you want
to do meta-optimization.

These routines can optimize functions whose inputs are structured
objects, such as dicts of numpy arrays."""

import taichi as ti
import taichi.math as tm

class Optim_template():
    def __init__(self, param_field):
        if(param_field.shape != ()):
            self.params = ti.field(dtype=model[0].param.dtype, shape=model.shape)
            self.grads = ti.field(dtype=model[0].param.dtype, shape=model.shape)
        else:
            self.params = ti.field(dtype=model[None].param.dtype, shape=())
            self.grads = ti.field(dtype=model[None].param.dtype, shape=())
        
    @ti.func
    def fetch_params_and_grads(self, model: ti.template()):
        if(model.shape != ()):
            for i in model:
                self.params[i] = model[i].param
                self.grads[i] = model[i].param.grad
        else:
            self.params[None] = model[None].param
            self.grads[None] = model[None].param.grad

    @ti.func
    def push_params(self, model: ti.template()):
        if(model.shape != ()):
            for i in model:
                model[i].param = self.params[i]
        else:
            model[None].param = self.params[None]

@ti.data_oriented
class SGD_Optimizer():
    def __init__(self, param_dict, step_size=0.1, mass=0.9):
        self.params = param_dict
        self.step_size = step_size
        self.mass = mass
        self.velocity = self.params.copy()
        for key in self.velocity:
            self.velocity[key] = ti.field(dtype=self.params[key].dtype, shape=self.params[key].shape)
            self.velocity[key].fill(0.0)

    @ti.kernel
    def _step_parm(self, param:ti.template(), velocity:ti.template()):
        for I in ti.grouped(param):
            velocity[I] = velocity[I] * self.mass - (1.0-self.mass) * param.grad[I]
            param[I] = param[I] + self.step_size * velocity[I]

    def step(self):
        for key in self.params:
            # print(self.params[key])
            # print(self.params[key].grad[None])
            self._step_parm(self.params[key], self.velocity[key])

@ti.data_oriented
class RMSProp_Optimizer():
    def __init__(self, param_field, step_size=0.1, gamma=0.9, eps=1e-8):
        self.params = param_field
        self.step_size = step_size
        self.gamma = gamma
        self.eps = eps
        self.avg_sq_grad = self.params.copy()
        for key in self.params:
            self.avg_sq_grad[key] = ti.field(dtype=self.params[key].dtype, shape=self.params[key].shape)
            self.avg_sq_grad[key].fill(0.0)

    @ti.kernel
    def _step_parm(self, param:ti.template(), avg_sq_grad:ti.template()):
        for I in ti.grouped(param):
            avg_sq_grad[I] = avg_sq_grad[I] * self.gamma + param.grad[I]**2 * (1 - self.gamma)
            param[I] = param[I] - self.step_size * param.grad[I] / (ti.sqrt(avg_sq_grad[I]) + self.eps)

    def step(self):
        for key in self.params:
            self._step_parm(self.params[key], self.avg_sq_grad[key])

@ti.data_oriented
class Adam_Optimizer():
    def __init__(self, param_field, step_size=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.params = param_field
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        self.m = self.params.copy()
        self.v = self.params.copy()
        self.iter = 1

        for key in self.params:
            self.m[key] = ti.field(dtype=self.params[key].dtype, shape=self.params[key].shape)
            self.v[key] = ti.field(dtype=self.params[key].dtype, shape=self.params[key].shape)
            self.m[key].fill(0.0)
            self.v[key].fill(0.0)
    
    @ti.kernel
    def _step_parm(self, param:ti.template(), m:ti.template(), v:ti.template()):
        for I in ti.grouped(param):
            m[I] = (1 - self.b1) * param.grad[I] + self.b1 * m[I]
            v[I] = (1 - self.b2) * (param.grad[I]**2) + self.b2 * v[I]
            mhat = m[I] / (1 - self.b1 ** self.iter)
            vhat = v[I] / (1 - self.b2 ** self.iter)
            param[I] = param[I] - self.step_size * mhat / (ti.sqrt(vhat) + self.eps)

    def step(self):
        for key in self.params:
            self._step_parm(self.params[key], self.m[key], self.v[key])


