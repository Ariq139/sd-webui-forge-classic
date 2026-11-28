"""Scoped constructor compatibility for the documented PrunaVAED architecture.

Algorithm reference: PrunaAI/P-LTX-2.3-VAED, patch_diffusers.py at ffee30b.
Uses the existing Diffusers 0.39.0 modules rather than executing downloaded code.
"""
import inspect
import threading
from contextlib import contextmanager
from functools import wraps

_LOCK = threading.RLock()
PRUNA_CHANNELS = (128, 256, 384, 1024)


def decoder_widths(channels, factors):
    if len(channels) != len(factors) or not channels:
        raise ValueError("Invalid decoder block layout.")
    current = channels[-1]
    result = []
    for nominal, factor in zip(reversed(channels), reversed(factors)):
        if factor < 1 or nominal % factor:
            raise ValueError("Invalid decoder channel factor.")
        out = nominal // factor
        result.append((current, nominal, out))
        current = out
    return result


@contextmanager
def pruna_construction(module, version, config):
    """Patch only construction of the known pruned layout, restoring on every exit."""
    channels = tuple(config.get("decoder_block_out_channels", ()))
    if channels != PRUNA_CHANNELS:
        yield
        return
    if version != "0.39.0" or config.get("timestep_conditioning", False):
        raise RuntimeError("PrunaVAED compatibility requires the verified 0.39.0 constructor and non-timestep-conditioned decoder.")
    widths = decoder_widths(channels, tuple(config["upsample_factor"]))
    decoder = module.LTX2VideoDecoder3d
    block = module.LTX2VideoUpBlock3d
    original_decoder, original_block = decoder.__init__, block.__init__
    sig = inspect.signature(original_block)
    with _LOCK:
        @wraps(original_decoder)
        def decoder_init(self, *args, **kwargs):
            bound = inspect.signature(original_decoder).bind(self, *args, **kwargs)
            bound.apply_defaults()
            if tuple(bound.arguments["block_out_channels"]) != channels:
                return original_decoder(self, *args, **kwargs)
            planned = iter(widths)

            @wraps(original_block)
            def block_init(instance, *a, **kw):
                values = sig.bind(instance, *a, **kw)
                values.apply_defaults()
                incoming, pre, out = next(planned)
                args_map = dict(values.arguments)
                args_map.pop("self")
                if args_map["out_channels"] != out or not args_map["spatio_temporal_scale"] or args_map["timestep_conditioning"]:
                    raise RuntimeError("Unexpected PrunaVAED decoder block layout.")
                args_map["in_channels"] = out
                original_block(instance, **args_map)
                if incoming != pre:
                    instance.conv_in = module.LTX2VideoResnetBlock3d(
                        in_channels=incoming, out_channels=pre, dropout=args_map["dropout"],
                        eps=args_map["resnet_eps"], non_linearity=args_map["resnet_act_fn"],
                        inject_noise=args_map["inject_noise"], timestep_conditioning=False,
                        spatial_padding_mode=args_map["spatial_padding_mode"])

            block.__init__ = block_init
            try:
                original_decoder(self, *args, **kwargs)
                if next(planned, None) is not None:
                    raise RuntimeError("Incomplete PrunaVAED decoder layout.")
            finally:
                block.__init__ = original_block

        decoder.__init__ = decoder_init
        try:
            yield
        finally:
            decoder.__init__ = original_decoder
