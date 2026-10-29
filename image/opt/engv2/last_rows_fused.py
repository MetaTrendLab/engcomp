import sys

O_KV_OLD = '''                        attn_output = self.attn(q, None, None)
                        output, _ = self.o_proj(attn_output)
                        return output
'''
O_KV_NEW = '''                        attn_output = self.attn(q, None, None)
                        return self._reax_o_rows(attn_output)
'''
O_OLD = '''                v = _rf.head_norm(v, None, self.num_kv_heads, self.head_dim, self.v_norm.variance_epsilon)
                attn_output = self.attn(q, k, v)
                output, _ = self.o_proj(attn_output)
                return output
'''
O_NEW = '''                v = _rf.head_norm(v, None, self.num_kv_heads, self.head_dim, self.v_norm.variance_epsilon)
                attn_output = self.attn(q, k, v)
                return self._reax_o_rows(attn_output)
'''
KVD_OLD = '''    def _reax_kv_direct(self):
'''
KVD_NEW = '''    def _reax_o_rows(self, attn_output):
        # last_rows_fused.py: the fused path's o_proj, on the sampled rows only in the last layer
        if _REAX_LAST_ROWS and getattr(self, "_reax_last", False):
            _rows = _reax_rows(attn_output)
            _o, _ = self.o_proj(attn_output.index_select(0, _rows))
            return _o.new_zeros((attn_output.shape[0], _o.shape[1])).index_copy(0, _rows, _o)
        output, _ = self.o_proj(attn_output)
        return output

''' + KVD_OLD
REST_OLD = '''    def _reax_fused_rest(self, attn_out, residual):
'''
REST_NEW = '''    def _reax_fused_rest(self, attn_out, residual):
        if _REAX_LAST_ROWS and getattr(self, "_reax_last", False):
            # last_rows_fused.py: the last layer's fused tail on the sampled rows; the final norm handed over the same way
            rows = _reax_rows(attn_out)
            out, _ = self._reax_fused_rest_all(attn_out.index_select(0, rows), residual.index_select(0, rows))
            n = _rf.take_prenorm(out)
            full = out.new_zeros((attn_out.shape[0], out.shape[1])).index_copy(0, rows, out)
            if n is not None:
                _rf.put_prenorm(full, n.new_zeros((attn_out.shape[0], n.shape[1])).index_copy(0, rows, n))
            return full, None
        return self._reax_fused_rest_all(attn_out, residual)

    def _reax_fused_rest_all(self, attn_out, residual):
'''
SET_OLD = '''    buf[:n].copy_(logits_indices, non_blocking=True)
'''
SET_NEW = '''    buf[:n].copy_(logits_indices, non_blocking=True)
    _REAX_ROWS["n"] = n
'''
ADD = '''

def _reax_rows(x):
    """last_rows_fused.py: eager callers get exactly this step's sampled rows (no padding to max_num_seqs); traced or
    captured code keeps the fixed-size reax::last_rows buffer."""
    n = _REAX_ROWS.get("n", 0)
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing() or not n:
        return torch.ops.reax.last_rows(x)
    return _REAX_ROWS["buf"][:n].clamp(0, max(x.shape[0] - 1, 0))
'''


if __name__ == "__main__":
    p = sys.argv[1]
    src = open(p).read()
    for old, new in ((O_KV_OLD, O_KV_NEW), (O_OLD, O_NEW), (KVD_OLD, KVD_NEW), (REST_OLD, REST_NEW), (SET_OLD, SET_NEW)):
        assert src.count(old) == 1, f"{p}: anchor found {src.count(old)} times: {old[:70]!r}"
        src = src.replace(old, new)
    open(p, "w").write(src + ADD)
    print("last_rows_fused: installed")
