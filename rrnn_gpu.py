import torch
from torch.autograd import Function
from collections import namedtuple
from pynvrtc.compiler import Program
from cupy.cuda import function
import numpy as np
from cuda.utils import *
from cuda.rrnn import *


class RRNN_Compute_GPU(Function):

    _RRNN_PROG = Program(UTIL + RRNN, "rrnn_prog.cu")
    _RRNN_PTX = _RRNN_PROG.compile()
    _DEVICE2FUNC = {}


    def __init__(self, d_out, k):
        super(RRNN_Compute_GPU, self).__init__()
        self.d_out = d_out
        self.k = k


    def compile_functions(self):
        device = torch.cuda.current_device()
        print ("RRNN loaded for gpu {}".format(device))
        mod = function.Module()
        mod.load(bytes(self._RRNN_PTX.encode()))

        fwd_func = mod.get_function("rrnn_fwd")
        bwd_func = mod.get_function("rrnn_bwd")
        Stream = namedtuple("Stream", ["ptr"])
        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self._DEVICE2FUNC[device] = (
            current_stream, fwd_func, bwd_func,
        )
        return current_stream, fwd_func, bwd_func


    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()


    def forward(self, u, c_init=None):
        assert u.size(-1) == self.k
        batch, length = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim
        thread_per_block = min(1024, ncols)
        num_block = (ncols-1)//thread_per_block+1
        if c_init is None:
            assert False

        size = (batch, length, dim)
        cs = u.new(*size)
        stream, fwd_func, _ = self.get_functions()
        FUNC = fwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            c_init.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            cs.data_ptr()],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        self.save_for_backward(u, c_init)
        self.intermediate_cs = cs
        last_c = cs[-1,...].view(batch, -1)
        return cs, last_c


    def backward(self, grad_cs, grad_last_c):
        u, c_init = self.saved_tensors
        cs = self.intermediate_cs
        batch, length = u.size(0), u.size(1)
        dim = self.d_out
        ncols = batch*dim
        thread_per_block = min(1024, ncols)
        num_block = (ncols-1)//thread_per_block+1

        if c_init is None:
            assert False
        # init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_init_c = u.new(batch, dim)
        stream, _, bwd_func = self.get_functions()
        FUNC = bwd_func

        FUNC(args=[
            u.contiguous().data_ptr(),
            c_init.contiguous().data_ptr(),
            cs.data_ptr(),
            grad_cs.data_ptr(),
            grad_last_c.contiguous().data_ptr(),
            np.int32(length),
            np.int32(batch),
            np.int32(dim),
            np.int32(self.k),
            grad_u.data_ptr(),
            grad_init_c.data_ptr()],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )
        return grad_u, grad_init_c
