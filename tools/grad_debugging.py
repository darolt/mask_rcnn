"""
This module is used for debugging backward gradients.

Licensed under The MIT License
Written by Jean Da Rolt
"""


def _get_printer(msg):
    """This function returns a printer function, that prints a message
    and then print a tensor. Used by register_hook in the backward pass.
    """
    def printer(tensor):
        """Closure function. See _get_printer docstring."""
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(f"{msg} shape: {tensor.shape}"
                  f" max: {tensor.abs().max()} min: {tensor.abs().min()}"
                  f" mean: {tensor.abs().mean()}")
    return printer


def register_hook(tensor, msg):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    # return
    if not tensor.requires_grad:
        print(f"Tensor does not require grad. ({msg})")
        return
    tensor.retain_grad()
    tensor.register_hook(_get_printer(msg))
