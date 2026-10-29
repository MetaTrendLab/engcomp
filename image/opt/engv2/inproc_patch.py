import sys

p = sys.argv[1]
s = open(p).read()
old = "        self.image_client = VllmClient(_img) if _img else self.client\n"
assert s.count(old) == 1, "image_route hunk missing (run image_route.py first)"
s = s.replace(old, old + '''        if os.environ.get("REAX_INPROC") == "1":  # engv2 inproc.py: the text instance runs inside this process
            from engine.inproc import install
            install(self, model_dir)
''')
open(p, "w").write(s)
print("inproc_patch: installed")
