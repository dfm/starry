# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from theano import gof
import theano.tensor as tt

__all__ = ["LinearOp"]


class LinearOp(tt.Op):

    def __init__(self, map):
        self.map = map
        self._grad_op = LinearGradientOp(self)

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        dt = inputs[-1].dtype
        outputs = [tt.TensorType(dt, (False, False))()]
        outputs += [tt.TensorType(dt, [False] * (inputs[0].ndim + 2))()]
        outputs += [tt.TensorType(dt, (False, False))() for _ in range(6)]
        # Note that the line above used to read:
        #   outputs = [inputs[-1].type()]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        out_shape = shapes[-1] + (tt.as_tensor(self.map.Ny),)
        grad_shapes = [shapes[0] + out_shape]
        grad_shapes += [out_shape for _ in range(6)]
        return [out_shape] + grad_shapes
        # Note that the line above used to read:
        #   return shapes[-1],

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    def perform(self, node, inputs, outputs):
        u, inc, obl, theta, xo, yo, zo, ro = inputs
        if self.map.udeg:
            self.map[1:] = u
        self.map.inc = inc
        self.map.obl = obl
        val, grad = self.map.linear_flux_model(
            theta=theta, xo=xo, yo=yo, zo=zo, ro=ro, gradient=True)

        outputs[0][0] = val

        # Save the gradients
        outputs[1][0] = grad["u"]
        outputs[2][0] = grad["inc"]
        outputs[3][0] = grad["obl"]
        outputs[4][0] = grad["theta"]
        outputs[5][0] = grad["xo"]
        outputs[6][0] = grad["yo"]
        outputs[7][0] = grad["ro"]

    def grad(self, inputs, gradients):
        bf = gradients[0]
        _, du, dinc, dobl, dtheta, dxo, dyo, dro = self(*inputs)
        return (
            tt.sum(du * bf, axis=(1, 2)),
            tt.sum(dinc * bf),
            tt.sum(dobl * bf),
            tt.sum(dtheta * bf, axis=-1),
            tt.sum(dxo * bf, axis=-1),
            tt.sum(dyo * bf, axis=-1),
            tt.zeros_like(inputs[6]),
            tt.sum(dro * bf, axis=-1),
        )


class LinearGradientOp(tt.Op):

    def __init__(self, base_op):
        self.base_op = base_op

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        outputs = [i.type() for i in inputs[:8]]
        return gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, shapes):
        return shapes[:8]

    def perform(self, node, inputs, outputs):
        u, inc, obl, theta, xo, yo, zo, ro, bf = inputs
        if self.base_op.map.udeg:
            self.base_op.map[1:] = u
        self.base_op.map.inc = inc
        self.base_op.map.obl = obl
        _, grad = self.base_op.map.linear_flux_model(
            theta=theta, xo=xo, yo=yo, zo=zo, ro=ro, gradient=True)

        for k, v in grad.items():
            print(k, v.shape)
        assert 0

        # Limb darkening gradient
        outputs[0][0] = np.array(np.sum(grad["u"] * bf, axis=(1, 2)))

        # Orientation gradients
        outputs[1][0] = np.atleast_1d(np.array(np.sum(grad["inc"] * bf)))
        outputs[2][0] = np.atleast_1d(np.array(np.sum(grad["obl"] * bf)))
        outputs[3][0] = np.array(np.sum(grad["theta"] * bf, axis=-1))

        # Occultation gradients
        outputs[4][0] = np.array(np.sum(grad["xo"] * bf, axis=-1))
        outputs[5][0] = np.array(np.sum(grad["yo"] * bf, axis=-1))
        outputs[6][0] = np.zeros_like(outputs[5][0])
        outputs[7][0] = np.array(np.sum(grad["ro"] * bf, axis=-1))

        # Reshape
        for i in range(8):
            outputs[i][0] = outputs[i][0].reshape(np.shape(inputs[i]))
