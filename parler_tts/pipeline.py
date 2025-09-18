"""
Enhanced pipeline support for ParlerTTS with modern transformers compatibility
"""

from typing import Any, Dict, List, Optional, Union
from transformers import Pipeline, AutoTokenizer
from transformers.utils import logging
import torch
import numpy as np

logger = logging.get_logger(__name__)

class ParlerTTSPipeline(Pipeline):
    """
    Text-to-Speech pipeline using ParlerTTS models with enhanced compatibility
    """
    
    def __init__(self, model, tokenizer=None, **kwargs):
        # Ensure model registration
        from . import register_parler_tts_models
        register_parler_tts_models()
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
            
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        generate_kwargs = {}
        postprocess_kwargs = {}
        
        # Handle common parameters
        if "description" in kwargs:
            preprocess_kwargs["description"] = kwargs["description"]
        if "max_length" in kwargs:
            generate_kwargs["max_length"] = kwargs["max_length"]
        if "temperature" in kwargs:
            generate_kwargs["temperature"] = kwargs["temperature"]
        if "do_sample" in kwargs:
            generate_kwargs["do_sample"] = kwargs["do_sample"]
            
        return preprocess_kwargs, generate_kwargs, postprocess_kwargs
    
    def preprocess(self, text, description=None):
        """Preprocess text and description for TTS generation"""
        if description is None:
            description = "A clear, pleasant voice speaks naturally."
            
        # Tokenize inputs
        description_tokens = self.tokenizer(description, return_tensors="pt", padding=True)
        prompt_tokens = self.tokenizer(text, return_tensors="pt", padding=True)
        
        return {
            "description_input_ids": description_tokens.input_ids,
            "description_attention_mask": description_tokens.attention_mask,
            "prompt_input_ids": prompt_tokens.input_ids,
            "prompt_attention_mask": prompt_tokens.attention_mask,
        }
    
    def _forward(self, model_inputs, **generate_kwargs):
        """Generate audio tokens using the model"""
        with torch.no_grad():
            generation = self.model.generate(
                input_ids=model_inputs["description_input_ids"],
                attention_mask=model_inputs["description_attention_mask"],
                prompt_input_ids=model_inputs["prompt_input_ids"],
                prompt_attention_mask=model_inputs["prompt_attention_mask"],
                **generate_kwargs
            )
        return {"generated_tokens": generation}
    
    def postprocess(self, model_outputs):
        """Convert generated tokens to audio waveform"""
        generated_tokens = model_outputs["generated_tokens"]
        
        # Decode using the model's audio decoder
        if hasattr(self.model, 'audio_encoder'):
            audio_values = self.model.audio_encoder.decode(
                generated_tokens,
                audio_scales=[None] * generated_tokens.shape[0]
            )
            # Convert to numpy for compatibility
            if isinstance(audio_values, torch.Tensor):
                audio_values = audio_values.cpu().numpy()
            return {"audio": audio_values[0], "sampling_rate": self.model.audio_encoder.sampling_rate}
        else:
            logger.warning("Audio encoder not found, returning raw tokens")
            return {"audio": generated_tokens.cpu().numpy(), "sampling_rate": 24000}

def create_parler_tts_pipeline(model_name_or_path="parler-tts/parler-tts-mini-v1", **kwargs):
    """
    Create a ParlerTTS pipeline with enhanced error handling
    """
    try:
        from .modeling_parler_tts import ParlerTTSForConditionalGeneration
        from . import register_parler_tts_models
        
        # Ensure models are registered
        register_parler_tts_models()
        
        # Load model and tokenizer
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        return ParlerTTSPipeline(model=model, tokenizer=tokenizer, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to create ParlerTTS pipeline: {e}")
        raise