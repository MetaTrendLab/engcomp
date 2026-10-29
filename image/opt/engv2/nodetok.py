import sys

ANCHOR = '''            request_id_item = f"{request_id}-{i}"
'''
NEW = '''            # engv2 nodetok.py (REAX_NODETOK=1): token-id logprob answers need no text in the output processor
            if (_REAX_NODETOK and isinstance(sampling_params, SamplingParams) and request.return_tokens_as_token_ids
                    and not request.echo and not request.stream and not sampling_params.stop):
                sampling_params.detokenize = False

'''
TAIL = '''

# ---------------------------------------------------------------- engv2 Blink: no detokenization for token-id answers
import os as _reax_os

_REAX_NODETOK = _reax_os.environ.get("REAX_NODETOK", "0") == "1"
'''

p = sys.argv[1]
s = open(p).read()
assert s.count(ANCHOR) == 1, "serving.py: request_id_item line not found"
assert "_REAX_NODETOK" not in s, "nodetok already installed"
s = s.replace(ANCHOR, NEW + ANCHOR) + TAIL
open(p, "w").write(s)
print("nodetok: installed")
