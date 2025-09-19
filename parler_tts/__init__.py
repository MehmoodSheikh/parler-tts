__version__ = "0.2.2"

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.utils import logging
from importlib.metadata import version
from packaging.version import Version
import warnings

# Import configurations first
from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .dac_wrapper import DACConfig, DACModel
from .modeling_parler_tts import (
    ParlerTTSForCausalLM,
    ParlerTTSForConditionalGeneration,
    apply_delay_pattern_mask,
    build_delay_pattern_mask,
)
from .streamer import ParlerTTSStreamer

logger = logging.get_logger(__name__)

# Check transformers version for compatibility
TRANSFORMERS_VERSION = Version(version("transformers"))
IS_TRANSFORMERS_4_44_PLUS = TRANSFORMERS_VERSION > Version("4.44.0")

def register_parler_tts_models():
    """Register ParlerTTS models with transformers AutoModel system"""
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        
        # Register configurations
        AutoConfig.register("parler_tts", ParlerTTSConfig)
        AutoConfig.register("parler_tts_decoder", ParlerTTSDecoderConfig)
        
        # Register models with both AutoModel classes
        AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
        AutoModelForSeq2SeqLM.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
        AutoModelForCausalLM.register(ParlerTTSConfig, ParlerTTSForCausalLM)
        
        # Register DAC if available
        if not is_dac_integrated_to_transformers:
            AutoConfig.register("dac", DACConfig)
        else:
            AutoConfig.register("dac_on_the_hub", DACConfig)
        AutoModel.register(DACConfig, DACModel)
        
        logger.info("✅ All ParlerTTS models registered successfully")
        
    except Exception as e:
        logger.warning(f"Model registration incomplete: {e}")
        # Don't fail completely, just warn

# Automatically register models on import
register_parler_tts_models()

# Import pipeline support
from .pipeline import ParlerTTSPipeline, create_parler_tts_pipeline

# Export main classes
__all__ = [
    "ParlerTTSConfig",
    "ParlerTTSDecoderConfig", 
    "ParlerTTSForCausalLM",
    "ParlerTTSForConditionalGeneration",
    "ParlerTTSStreamer",
    "DACConfig",
    "DACModel",
    "apply_delay_pattern_mask",
    "build_delay_pattern_mask",
    "register_parler_tts_models",
    "ParlerTTSPipeline",
    "create_parler_tts_pipeline"
]
