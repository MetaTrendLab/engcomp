import sys

p = sys.argv[1]
s = open(p).read()
old = '''        self.client = VllmClient(base_url, completion_temperature=1.0 if family == "quyet" and self.preset.name == "ref-prod" else 0)
'''
new = old + '''        _img = os.environ.get("REAX_VLLM_IMAGE_URL")
        self.image_client = VllmClient(_img) if _img else self.client
'''
assert s.count(old) == 1, "client hunk"
s = s.replace(old, new)
old = "            values, e.n_tokens = self.client.image_option_values(it.messages, it.letter_ids)\n"
assert s.count(old) == 1, "image hunk"
s = s.replace(old, old.replace("self.client.", "self.image_client."))
open(p, "w").write(s)
print("image_route: installed")
