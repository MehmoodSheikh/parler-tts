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
    """Register ParlerTTS models with transformers AutoModel classes"""
    try:
        # Register ParlerTTS configurations
        if not hasattr(AutoConfig, "_name_to_config") or "parler_tts" not in AutoConfig._name_to_config:
            AutoConfig.register("parler_tts", ParlerTTSConfig)
            logger.info("Registered ParlerTTSConfig")
        
        if not hasattr(AutoConfig, "_name_to_config") or "parler_tts_decoder" not in AutoConfig._name_to_config:
            AutoConfig.register("parler_tts_decoder", ParlerTTSDecoderConfig)
            logger.info("Registered ParlerTTSDecoderConfig")
        
        # Register ParlerTTS models
        AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration, exist_ok=True)
        AutoModelForSeq2SeqLM.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration, exist_ok=True)
        AutoModelForCausalLM.register(ParlerTTSDecoderConfig, ParlerTTSForCausalLM, exist_ok=True)
        
        # Handle DAC model registration based on transformers version
        dac_config_name = "dac_on_the_hub" if IS_TRANSFORMERS_4_44_PLUS else "dac"
        
        if not hasattr(AutoConfig, "_name_to_config") or dac_config_name not in AutoConfig._name_to_config:
            AutoConfig.register(dac_config_name, DACConfig)
            logger.info(f"Registered DACConfig as {dac_config_name}")
        
        AutoModel.register(DACConfig, DACModel, exist_ok=True)
        
        logger.info("All ParlerTTS models registered successfully")
        
    except Exception as e:
        logger.warning(f"Failed to register some ParlerTTS models: {e}")
        # Continue anyway - the models might still work

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
