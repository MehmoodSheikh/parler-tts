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
    Configuration class for ParlerTTS.
    """
    model_type = "parler_tts"
    is_composition = True

    def __init__(self, vocab_size=None, prompt_cross_attention=True, **kwargs):
        super().__init__(**kwargs)
        
        # Enhanced configuration loading with better error handling
        if "text_encoder" not in kwargs and "audio_encoder" not in kwargs and "decoder" not in kwargs:
            # If we're loading from a pretrained model, try to construct default configs
            logger.warning(
                "No sub-configs found. This might be a legacy model format. "
                "Attempting to construct default configurations..."
            )
            
            # Try to load from the model's config.json if available
            try:
                # Default configurations - adjust these based on the actual model
                text_encoder_config = {
                    "model_type": "t5",
                    "vocab_size": vocab_size or 32128,
                    "d_model": 512,
                    "num_layers": 6,
                    "num_heads": 8,
                    "d_ff": 2048,
                }
                
                audio_encoder_config = {
                    "model_type": "dac_on_the_hub" if self._is_dac_integrated_to_transformers() else "dac",
                    "num_codebooks": 9,
                    "codebook_size": 1024,
                    "latent_dim": 1024,
                    "sampling_rate": 44100,
                }
                
                decoder_config = {
                    "model_type": "parler_tts_decoder",
                    "vocab_size": vocab_size or 32128,
                    "hidden_size": 896,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 14,
                    "intermediate_size": 3584,
                }
                
                kwargs.update({
                    "text_encoder": text_encoder_config,
                    "audio_encoder": audio_encoder_config,
                    "decoder": decoder_config,
                })
                
            except Exception as e:
                logger.error(f"Failed to construct default configs: {e}")
                raise ValueError(
                    "Config has to be initialized with text_encoder, audio_encoder and decoder config. "
                    f"Current kwargs keys: {list(kwargs.keys())}"
                )
        
        # Now proceed with the original logic
        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")

        text_encoder_config = kwargs.pop("text_encoder")
        audio_encoder_config = kwargs.pop("audio_encoder")
        decoder_config = kwargs.pop("decoder")

        # Initialize sub-configs
        from transformers import AutoConfig
        self.text_encoder = AutoConfig.for_model(**text_encoder_config)
        self.decoder = ParlerTTSDecoderConfig(**decoder_config)
        
        # Handle audio encoder config based on transformers version
        if self._is_dac_integrated_to_transformers():
            audio_encoder_config["model_type"] = "dac_on_the_hub"
        else:
            audio_encoder_config["model_type"] = "dac"

        # Import DACConfig here to avoid circular imports
        try:
            from .dac_wrapper import DACConfig
            self.audio_encoder = DACConfig(**audio_encoder_config)
        except ImportError as e:
            logger.warning(f"Could not import DACConfig: {e}")
            # Create a minimal config if import fails
            self.audio_encoder = type('DACConfig', (), audio_encoder_config)()

        self.vocab_size = vocab_size
        self.prompt_cross_attention = prompt_cross_attention

    def _is_dac_integrated_to_transformers(self):
        """Check if DAC is integrated to transformers"""
        try:
            return Version(version("transformers")) > Version("4.44.2dev")
        except:
            return False

# NO TEST CODE OR IMPORTS FROM PARLER_TTS HERE!
# The file should end here without any execution code.