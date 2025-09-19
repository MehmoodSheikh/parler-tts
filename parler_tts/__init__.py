__version__ = "0.2.2"

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from importlib.metadata import version
from packaging.version import Version
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Define the missing variable
is_dac_integrated_to_transformers = Version(version("transformers")) > Version("4.44.2dev")

# Import configurations first
from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .dac_wrapper import DACConfig, DACModel

# Import model classes
from .modeling_parler_tts import (
    ParlerTTSForConditionalGeneration,
    ParlerTTSForCausalLM,
)

# Import other components
from .streamer import ParlerTTSStreamer

def register_parler_tts_models():
    """Register ParlerTTS models with transformers AutoModel system"""
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        
        # Register configurations
        AutoConfig.register("parler_tts", ParlerTTSConfig)
        AutoConfig.register("parler_tts_decoder", ParlerTTSDecoderConfig)
        
        # Register DAC configuration
        if not is_dac_integrated_to_transformers:
            AutoConfig.register("dac", DACConfig)
        else:
            AutoConfig.register("dac_on_the_hub", DACConfig)
        
        # Register models
        AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
        AutoModelForSeq2SeqLM.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
        AutoModelForCausalLM.register(ParlerTTSConfig, ParlerTTSForCausalLM)
        AutoModel.register(DACConfig, DACModel)
        
        logger.info("✅ All ParlerTTS models registered successfully")
        
    except Exception as e:
        logger.warning(f"Model registration incomplete: {e}")

# Auto-register on import
register_parler_tts_models()

__all__ = [
    "ParlerTTSConfig",
    "ParlerTTSDecoderConfig", 
    "ParlerTTSForConditionalGeneration",
    "ParlerTTSForCausalLM",
    "ParlerTTSStreamer",
    "DACConfig",
    "DACModel",
    "register_parler_tts_models",
    "is_dac_integrated_to_transformers",
]
