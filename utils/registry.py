_REG = {}
def register(name): return lambda fn: _REG.setdefault(name, fn)
def get(name): return _REG[name]
