RRNN = """

extern "C" {
     __global__ void rrnn_fwd(
                const float * __restrict__ u, 
                const float * __restrict__ c_init,
                const int len, 
                const int batch,
                const int dim,
                const int k,
                float * __restrict__ c) {
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;

        const float *up = u + (col*k);
        float *cp = c + col;
        float cur_c = *(c_init + col);

        for (int row = 0; row < len; ++row) {
            float u1 = *(up);

            float forget1 = *(up+1);

            cur_c = cur_c * forget1 + u1;

            *cp = cur_c;

            up += ncols_u;
            cp += ncols;
        }
    }

    __global__ void rrnn_bwd(
                const float * __restrict__ u, 
                const float * __restrict__ c_init,
                const float * __restrict__ c,
                const float * __restrict__ grad_c, 
                const float * __restrict__ grad_last_c,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ grad_u, 
                float * __restrict__ grad_c_init) {
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;

        float cur_c = *(grad_last_c + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *cp = c + col + (len-1)*ncols;

        const float *gcp = grad_c + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;

        for (int row = len-1; row >= 0; --row) {
            float u1 = *(up);
            float forget1 = *(up+1);

            const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(c_init+col));

            const float gc = *(gcp) + cur_c;

            float gu1 = gc;
            *(gup) = gu1;
            float gforget1 = gc*prev_c_val;
            *(gup+1) = gforget1;

            cur_c = gc * forget1;

            up -= ncols_u; 
            cp -= ncols;
            gup -= ncols_u;
            gcp -= ncols;
        }

        *(grad_c_init + col) = cur_c;
    }
}
"""