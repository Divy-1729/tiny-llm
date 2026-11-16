import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int, #D
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        angular_rate = mx.power(base, mx.arange(0, self.dims//2) / (-1*self.dims//2))
        positions = mx.arange(0, seq_len, dtype = mx.float32)
        positional_angles = mx.outer(positions, angular_rate)
        self.positional_sines, self.positional_cosines = mx.sin(positional_angles), mx.cos(positional_angles)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        input_dtype = x.dtype
        
        if self.traditional:
            x = x.reshape(*x.shape[:-1], x.shape[-1]//2, 2) #[N,L,H,M,2]
            a = x[:,:,:,:,0] # [N,L,H,M]
            b = x[:,:,:,:,1] #[N,L,H,M]
        else:
            a = x[:,:,:,:x.shape[-1]//2]
            b = x[:,:,:,x.shape[-1]//2 : ]
        L = a.shape[-3]
        if offset is None:
            c = self.positional_cosines[:L] #[L,M]
            s = self.positional_sines[:L] #[L,M]
        else:
              c = self.positional_cosines[offset] #[L,M]
              s = self.positional_sines[offset] #[L,M]
        cos = mx.reshape(c,shape = (1, c.shape[0], 1, c.shape[1])) #[1,L,1,M]
        sin = mx.reshape(s,shape = (1,s.shape[0],1,s.shape[1]))  #[1,L,1,M]
        a_rot = a*cos - b*sin # [ N, L, H, M]
        b_rot = a*sin + b*cos # [ N, L, H, M]
        if self.traditional:
            stacked = mx.stack([a_rot,b_rot],axis = -1) # [ N, L, H, M,2]
            output = stacked.reshape(*stacked.shape[:-2], stacked.shape[-2]*2) # [ N, L, H, D]
        else:
            output = mx.concatenate([a_rot,b_rot], axis = -1)
        return output.astype(input_dtype)

            



