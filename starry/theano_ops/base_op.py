# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StarryBaseOp"]

import os
import theano
from theano import gof
import theano.tensor as tt


class StarryBaseOp(gof.COp):

    __props__ = ()
    num_input = 0
    output_ndim = ()
    func_file = None
    func_name = None

    def __init__(self):
        super(StarryBaseOp, self).__init__(self.func_file, self.func_name)

    # def c_code_cache_version(self):
    #     return (0, 0, 1)

    def c_headers(self, compiler):
        return ["theano_helpers.h"]

    def c_header_dirs(self, compiler):
        thisdir = os.path.dirname(os.path.abspath(__file__))
        return [
            thisdir,
            os.path.dirname(thisdir),
            os.path.join(os.path.dirname(os.path.dirname(thisdir)),
                         "lib", "eigen_3.3.3"),
        ]

    def c_compile_args(self, compiler):
        args = ["-O2", "-DNDEBUG", "-std=c++14"]
        return args

    def make_node(self, *args):
        if len(args) != self.num_input:
            raise ValueError("expected {0} inputs".format(self.num_input))
        dtype = theano.config.floatX
        in_args = []
        for a in args:
            try:
                a = tt.as_tensor_variable(a)
            except tt.AsTensorError:
                pass
            else:
                dtype = theano.scalar.upcast(dtype, a.dtype)
            in_args.append(a)
        out_args = [
            tt.TensorType(dtype=dtype, broadcastable=[False] * ndim)()
            for ndim in self.output_ndim]
        return gof.Apply(self, in_args, out_args)
