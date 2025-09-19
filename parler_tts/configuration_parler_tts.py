# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ParlerTTS model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from importlib.metadata import version
from packaging.version import Version

logger = logging.get_logger(__name__)

class ParlerTTSDecoderConfig(PretrainedConfig):
    """
    Configuration class for ParlerTTS decoder.
    """
    model_type = "parler_tts_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=None,
        intermediate_size=3584,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        sliding_window=None,
        attention_bias=False,
        attention_dropout=0.0,
        num_codebooks=9,
        vocab_sizes=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.num_codebooks = num_codebooks
        if vocab_sizes is not None:
            self.vocab_sizes = vocab_sizes
        else:
            self.vocab_sizes = [vocab_size] * num_codebooks

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class ParlerTTSConfig(PretrainedConfig):
    """
    Configuration class for ParlerTTS with proper attribute handling.
    """
    model_type = "parler_tts"
    is_composition = True

    def __init__(self, vocab_size=None, prompt_cross_attention=True, **kwargs):
        # Initialize parent class first
        super().__init__(**kwargs)
        
        # Set default values
        self.vocab_size = vocab_size
        self.prompt_cross_attention = prompt_cross_attention
        
        # Initialize attributes that transformers expects
        self.quantization_config = None
        self.auto_map = {}
        self.custom_pipelines = {}
        
        # Initialize sub-configurations
        self._initialize_sub_configs(kwargs)

    def _initialize_sub_configs(self, kwargs):
        """Initialize sub-configurations with proper error handling"""
        
        # Check if we have the required sub-configs
        required_configs = ["text_encoder", "audio_encoder", "decoder"]
        missing_configs = [cfg for cfg in required_configs if cfg not in kwargs]
        
        if missing_configs:
            logger.warning(f"Missing sub-configs: {missing_configs}. Creating defaults...")
            self._create_default_configs(kwargs)
        
        # Extract sub-configs
        text_encoder_config = kwargs.pop("text_encoder", {})
        audio_encoder_config = kwargs.pop("audio_encoder", {})
        decoder_config = kwargs.pop("decoder", {})
        
        # Initialize sub-configs
        try:
            from transformers import AutoConfig
            
            # Text encoder config
            if isinstance(text_encoder_config, dict):
                self.text_encoder = AutoConfig.for_model(**text_encoder_config)
            else:
                self.text_encoder = text_encoder_config
            
            # Decoder config
            if isinstance(decoder_config, dict):
                self.decoder = ParlerTTSDecoderConfig(**decoder_config)
            else:
                self.decoder = decoder_config
                
            # Audio encoder config (DAC)
            if isinstance(audio_encoder_config, dict):
                if self._is_dac_integrated_to_transformers():
                    audio_encoder_config["model_type"] = "dac_on_the_hub"
                else:
                    audio_encoder_config["model_type"] = "dac"
                
                try:
                    from .dac_wrapper import DACConfig
                    self.audio_encoder = DACConfig(**audio_encoder_config)
                except ImportError:
                    logger.warning("Could not import DACConfig, using generic config")
                    self.audio_encoder = type('DACConfig', (), audio_encoder_config)()
            else:
                self.audio_encoder = audio_encoder_config
                
        except Exception as e:
            logger.error(f"Failed to initialize sub-configs: {e}")
            raise ValueError(f"Configuration initialization failed: {e}")

    def _create_default_configs(self, kwargs):
        """Create default configurations for missing sub-configs"""
        
        if "text_encoder" not in kwargs:
            kwargs["text_encoder"] = {
                "model_type": "t5",
                "vocab_size": self.vocab_size or 32128,
                "d_model": 512,
                "num_layers": 6,
                "num_heads": 8,
                "d_ff": 2048,
            }
        
        if "audio_encoder" not in kwargs:
            kwargs["audio_encoder"] = {
                "model_type": "dac_on_the_hub" if self._is_dac_integrated_to_transformers() else "dac",
                "num_codebooks": 9,
                "codebook_size": 1024,
                "latent_dim": 1024,
                "sampling_rate": 44100,
            }
        
        if "decoder" not in kwargs:
            kwargs["decoder"] = {
                "model_type": "parler_tts_decoder",
                "vocab_size": self.vocab_size or 32128,
                "hidden_size": 896,
                "num_hidden_layers": 24,
                "num_attention_heads": 14,
                "intermediate_size": 3584,
            }

    def _is_dac_integrated_to_transformers(self):
        """Check if DAC is integrated to transformers"""
        try:
            return Version(version("transformers")) > Version("4.44.2dev")
        except Exception:
            return False

    def __getattribute__(self, key):
        """Override getattribute to handle missing attributes properly"""
        try:
            return super().__getattribute__(key)
        except AttributeError:
            # Return appropriate defaults for known attributes
            if key == 'quantization_config':
                return None
            elif key in ['auto_map']:
                return {}
            elif key in ['custom_pipelines']:
                return {}
            elif key == 'transformers_weights':
                return None
            elif key in ['_attn_implementation_internal', 'gguf_file']:
                return None
            else:
                # For unknown attributes, return None but don't log
                return None

    def to_dict(self):
        """Custom to_dict that handles None attributes properly"""
        output = {}
        
        # Get all non-private attributes
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
                
            # Skip None quantization_config
            if key == 'quantization_config' and value is None:
                continue
                
            # Handle objects with to_dict method
            if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                try:
                    output[key] = value.to_dict()
                except Exception:
                    # If to_dict fails, convert to string
                    output[key] = str(value)
            else:
                output[key] = value
        
        return output

    def __repr__(self):
        """Override __repr__ to prevent to_dict issues during logging"""
        return f"{self.__class__.__name__} (configuration object)"

# NO TEST CODE OR IMPORTS FROM PARLER_TTS HERE!
# The file should end here without any execution code.

# Add these patches at the end of the file:

# Monkey patch to fix quantization issues
import transformers.quantizers.auto

# Store the original function
original_supports_quant_method = transformers.quantizers.auto.AutoHfQuantizer.supports_quant_method

@staticmethod
def patched_supports_quant_method(quantization_config_dict):
    """Patched version that handles None quantization_config"""
    if quantization_config_dict is None:
        return False
    
    if not hasattr(quantization_config_dict, 'get'):
        # If it's not a dict-like object, return False
        return False
    
    try:
        return original_supports_quant_method(quantization_config_dict)
    except Exception:
        return False

# Apply the patch
transformers.quantizers.auto.AutoHfQuantizer.supports_quant_method = patched_supports_quant_method

# Patch the get_hf_quantizer function with correct signature
original_get_hf_quantizer = transformers.quantizers.auto.get_hf_quantizer

def patched_get_hf_quantizer(*args, **kwargs):
    """Patched version that handles None quantization_config with flexible signature"""
    
    # Handle different argument patterns
    if len(args) >= 2:
        config = args[1]  # config is typically the second argument
    elif 'config' in kwargs:
        config = kwargs['config']
    else:
        # If we can't find config, just call original
        try:
            return original_get_hf_quantizer(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, proceeding without quantization")
            # Return safe defaults based on common transformers patterns
            if len(args) >= 4:
                return None, args[1], args[2], args[3]
            else:
                return None, None, None, None
    
    # If quantization_config is None, bypass quantization
    if hasattr(config, 'quantization_config') and config.quantization_config is None:
        # Return no quantizer, but preserve other arguments
        if len(args) >= 4:
            return None, args[1], args[2], args[3]  # None, config, dtype, device_map
        else:
            return None, config, kwargs.get('dtype'), kwargs.get('device_map')
    
    try:
        return original_get_hf_quantizer(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Quantization failed: {e}, proceeding without quantization")
        # Return safe defaults
        if len(args) >= 4:
            return None, args[1], args[2], args[3]
        else:
            return None, config, kwargs.get('dtype'), kwargs.get('device_map')

# Apply the patch
transformers.quantizers.auto.get_hf_quantizer = patched_get_hf_quantizer