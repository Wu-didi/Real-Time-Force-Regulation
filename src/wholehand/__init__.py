"""Contact perception and independent force-allocation reproduction components."""
__all__ = ["ContactSet", "ForceAllocator", "ForceSettings"]
def __getattr__(name):
    if name in __all__:
        from . import allocator
        return getattr(allocator, name)
    raise AttributeError(name)
