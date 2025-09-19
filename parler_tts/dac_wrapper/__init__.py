from .configuration_dac import DACConfig

try:
    from .modeling_dac import DACModel
    DAC_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"DACModel is not available due to missing dependencies: {e}. "
        "Please install descript-audio-codec: `pip install descript-audio-codec`"
    )
    DACModel = None
    DAC_AVAILABLE = False

__all__ = ["DACConfig"]
if DAC_AVAILABLE:
    __all__.append("DACModel")
