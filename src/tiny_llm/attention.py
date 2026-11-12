import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array, # N x L x D
    key: mx.array, # N x L x D -> transposed # N x D x L 
    value: mx.array, # N x L x D
    scale: float | None = None, # scalar
    mask: mx.array | None = None, # same as Q.K^T so N x L x L
) -> mx.array:
    if scale is not None:
        scalar = scale
    else: 
        scalar = 1/(mx.sqrt(mx.array(key.shape[-1])).item())
    if mask is not None:
        return mx.matmul(softmax(mx.matmul(query,key.swapaxes(-1,-2)) * scalar + mask, axis = -1),value)
    return  mx.matmul(softmax(mx.matmul(query,key.swapaxes(-1,-2)) * scalar, axis = -1),value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int, # D*H
        num_heads: int, # H
        wq: mx.array, # (H x D) x E
        wk: mx.array, # (H x D) x E
        wv: mx.array, # (H x D) x E
        wo: mx.array, # E x (H x D)
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array, # N x L x E
        key: mx.array,  # N x L x E
        value: mx.array, # N x L x E
        mask: mx.array | None = None, 
    ) -> mx.array:
        query_result = mx.matmul(query,self.wq.T) # N x L X (H.D)
        key_result = mx.matmul(key, self.wk.T) # N x L X (H.D)
        value_result = mx.matmul(value, self.wv.T) # N x L X (H.D)
 
        query_result=query_result.reshape(*query_result.shape[:-1], self.num_heads, self.hidden_size//self.num_heads) # N x L x H x D
        key_result=key_result.reshape(*key_result.shape[:-1], self.num_heads, self.hidden_size//self.num_heads) # N x L x H x D
        value_result=value_result.reshape(*value_result.shape[:-1], self.num_heads, self.hidden_size//self.num_heads) # N x L x H x D

        query_result=query_result.swapaxes(-2,-3) # N x H x L x D
        key_result=key_result.swapaxes(-2,-3) # N x H x L x D
        value_result=value_result.swapaxes(-2,-3) # N x H x L x D
        head = []
        for i in range(self.num_heads):
            head_i = scaled_dot_product_attention_simple(query_result[:,i, :, :], key_result[:,i, :, :], value_result[:,i, :, :], mask=mask) #(N x L x D)
            head.append(head_i)
        mla = mx.stack(head, axis = -1) # (N x L x D x H)
        mla = mla.swapaxes(-1,-2) # (N x L x D x H) -> N x L x H x D
        mla = mla.reshape(*mla.shape[:-2], self.hidden_size) # N x L x H x D -> N x L x (H.D)

        return mx.matmul(mla, self.wo.T)





    

        


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Attend to paged K/V storage without reconstructing a dense cache.

    Week 3 Day 4 owns the correctness-first decode and long-prefill paths for
    both float32 and BF16. Day 5 optimizes the same public boundary.
    """
    pass
