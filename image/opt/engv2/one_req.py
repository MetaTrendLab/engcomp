import sys

OLD = '''            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )
'''
NEW = '''            else:
                # engv2 one_req.py: a REAX wave in one request, each prompt with its own option letters
                sampling_params = (_reax_one_req(request, i) if request.vllm_xargs
                                   and "reax_allowed" in request.vllm_xargs else request).to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )
'''
ADD = '''


# ---------------------------------------------------------------- engv2 Blink: one request per REAX wave (one_req.py)
def _reax_one_req(request, i):
    """A copy of the request for prompt i: its own allowed_token_ids and logprobs, vllm_xargs without reax_allowed."""
    xa = dict(request.vllm_xargs)
    sets = str(xa.pop("reax_allowed")).split(";")
    allowed = [int(t) for t in sets[i].split(",")]
    return request.model_copy(update={"allowed_token_ids": allowed, "logprobs": len(allowed), "vllm_xargs": xa or None})
'''

path = sys.argv[1]
src = open(path).read()
assert src.count(OLD) == 1, f"serving.py: to_sampling_params anchor found {src.count(OLD)} times"
open(path, "w").write(src.replace(OLD, NEW) + ADD)
print("one_req: installed")
