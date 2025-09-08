# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from megatron.energon import VQASample, batch_list, batch_pad_stack
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample, MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder, VQASampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.utils import logging


@dataclass
class LlavaNextTextSample(ImageTextSample):
    '''
    Sample type for LLaVA-Next, extending ImageTextSample to support tiled image data.

    This class adds additional attributes for handling high-resolution images processed as tiles,
    along with metadata about the tiled images.

    Attributes:
        num_media_tiles (int): The number of tiles used to represent the high-resolution image.
        image_sizes (torch.Tensor): A tensor representing the sizes of the tiled images.
        attention_mask (Optional[torch.Tensor]): An optional attention mask for the sample,
        used to determine which tokens or tiles are attended to during processing. Defaults to None.
    '''

    num_media_tiles: int = 0
    image_sizes: torch.tensor = field(default_factory=lambda: torch.tensor([]))
    attention_mask: Optional[torch.tensor] = None


@dataclass
class LlavaNextTextRawBatch(ImageTextRawBatch):
    """
    Batch type for raw LLaVA-Next samples, supporting tiled image data.

    This class aggregates multiple `LlavaNextTextSample` instances into a batch for processing.
    It includes attributes for managing tiled images and associated metadata for each sample in the batch.

    Attributes:
        num_media_tiles (List[int]): A list containing the number of tiles for each image in the batch.
        image_sizes (torch.Tensor): A tensor containing the sizes of all tiled images in the batch.
        attention_mask (Optional[torch.Tensor]): Attention mask. Defaults to None.
    """

    num_media_tiles: List[int] = field(default_factory=list)
    image_sizes: torch.tensor = field(default_factory=lambda: torch.tensor([]))
    attention_mask: Optional[torch.tensor] = None


class LlavaNextSampleEncoder(VQASampleEncoder):
    """LlavaNextSampleEncoder"""

    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):
        """
        Initialize the LlavaNextSampleEncoder, inherited from VQASampleEncoder for multimodal samples
        focused on VQA-style data to support LLaVANeXT

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MultiModalSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)

    def process_image(self, image):
        """
        Process and prepare an image sample for encoding.

        This method preprocesses the image using the HF image_processor, converting it to
        a tensor.

        Parameters:
        image: The input image to be processed.

        Returns:
        torch.Tensor: The processed image tensor.
        """
        image_array = self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        return image_array

    def encode(self, input_sample: VQASample, output_sample: LlavaNextTextSample):
        """
        Encode a single sample into a format suitable for model input.

        This method prepares the conversation prompt, tokenizes it, and processes
        the associated image. It fills the output sample with tokens, labels, loss mask,
        and other required fields for multimodal processing.

        Parameters:
        input_sample (VQASample): The input VQA sample containing an image and conversation text.
        output_sample (LlavaNextTextSample): The output sample structure where encoded results are stored.

        Returns:
        LlavaNextTextSample: The encoded output sample, containing processed tokens, labels,
            images, loss masks, and metadata.
        """
        conversation_prompt = self.apply_prompt_template(input_sample)
        logging.debug(f"[Energon] task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        tokens = self.tokenize(conversation_prompt)
        labels = self.compute_labels(tokens, input_sample)
        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        logging.debug(f"[Energon] task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"[Energon] task encoder encode_sample lables {labels}")
        loss_mask = self.compute_loss_mask(labels)
        processed_image = self.process_image(input_sample.image)
        output_sample.__key__ = input_sample.__key__
        output_sample.images = processed_image
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.num_media_tiles = processed_image.shape[0]
        output_sample.attention_mask = torch.ones(len(tokens), dtype=torch.long)
        height = input_sample.image.shape[1]
        width = input_sample.image.shape[2]
        output_sample.image_sizes = torch.tensor([[height, width]], dtype=torch.long)
        return output_sample


class LlavaNextTaskEncoder(MultiModalTaskEncoder):
    """LlavaNextTaskEncoder"""

    def __init__(self, tokenizer, image_processor, multimodal_sample_config):
        """
        Initialize the LlavaNextTaskEncoder.

        This encoder extends MultiModalTaskEncoder to specifically handle LlavaNeXT,
        overriding  encoders for VQA sample type.

        Parameters:
        tokenizer (Tokenizer): The tokenizer for processing text data across sample types.
        image_processor (ImageProcessor): The image processor for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig): Configuration settings for multimodal samples.
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.encoders: Dict[str, SampleEncoder] = {
            VQASample.__name__: LlavaNextSampleEncoder(tokenizer, image_processor, multimodal_sample_config)
        }

    def batch(self, samples: List[LlavaNextTextSample]) -> LlavaNextTextRawBatch:
        """
        Batch multiple encoded samples into a single batch structure for model input.

        This method combines individual sample fields (keys, images, tokens, labels, etc.) and
        pads or stacks them as needed to create a unified batch.

        Parameters:
        samples (List[LlavaNextTextSample]): A list of LlavaNextTextSample instances to be batched.

        Returns:
        LlavaNextTextRawBatch: A batch containing all input samples' images, tokens, labels,
            loss masks, and other metadata prepared for model processing.
        """
        keys, images, tokens, labels, loss_mask, num_media_tiles, image_sizes, attention_mask = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for sample in samples:
            keys.append(sample.__key__)
            images.append(sample.images)
            tokens.append(sample.tokens)
            labels.append(sample.labels)
            loss_mask.append(sample.loss_mask)
            num_media_tiles.append(sample.num_media_tiles)
            image_sizes.append(sample.image_sizes)
            attention_mask.append(sample.attention_mask)

        batch_keys = batch_list(keys)

        batch_images = torch.cat(images, dim=0)

        batch_tokens = pad_sequence(tokens, batch_first=True)
        batch_labels = pad_sequence(labels, batch_first=True)
        image_sizes = torch.cat(image_sizes, dim=0)
        batch_loss_mask = batch_pad_stack(loss_mask)
        batch_attention_mask = batch_pad_stack(attention_mask)
        batch_num_media_tiles = torch.tensor(batch_list(num_media_tiles), dtype=torch.int)
        return LlavaNextTextRawBatch(
            __keys__=batch_keys,
            images=batch_images,
            tokens=batch_tokens,
            labels=batch_labels,
            loss_mask=batch_loss_mask,
            num_media_tiles=batch_num_media_tiles,
            image_sizes=image_sizes,
            attention_mask=batch_attention_mask,
        )
