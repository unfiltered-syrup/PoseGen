# PoseGen Project Report

## 1. Project Summary

PoseGen is a sprite animation generation project that produces short 2D pixel-art animation loops from a single input frame. The main use case is a game-editor workflow: a developer supplies one static character sprite, selects an animation row such as walking, casting, slashing, or drawing a bow, and receives the remaining frames needed for that animation strip.

Although the repository includes experiments with deterministic warping, an MSE baseline, a ConvGRU variant, and Wan2.2 image-to-video preparation, the central model in the current project is a learned patch-based sequence-to-sequence transformer. This transformer is designed specifically for Universal LPC-style sprites: low-resolution 64 x 64 pixel-art frames with transparent backgrounds, hard edges, repeated animation cycles, and row-structured action labels.

The transformer approach sits between two extremes. It is much more flexible than a hand-authored geometric warp because it can learn row-specific motion patterns from data, but it is far lighter and more controllable than a large general image-to-video model. The model is trained directly on composited sprite sheets, so its prior is aligned with pixel-art characters rather than natural video.

## 2. Dataset and Supervision

The dataset is generated from the Universal LPC spritesheet assets by `create_dataset.py`. Instead of relying on a small fixed set of characters, the script builds synthetic character sheets by randomly composing compatible sprite layers. Each generated character is assembled from body, clothing, hair, accessories, facial features, weapons, and related layers. The layer order is fixed so that assets appear in a visually plausible stack, from the base body outward to weapons and accessories.

The data generation process creates n composited character entries:

- n/2 male entries.
- n/2 female entries.
- One full spritesheet per entry.
- One metadata JSON file per entry recording which asset layers were used.
- One extracted PNG strip per animation row.

Each row strip is 64 pixels tall. Its width is a multiple of 64, where each 64 x 64 segment is one animation frame. The training dataset therefore treats every row as an independent supervised sequence:

- `frame0`: the first frame in the row, used as the conditioning input.
- `target_frames`: all remaining frames in that row.
- `row_label`: the integer row index, used to tell the model which animation type to generate.

This setup matches the structure of LPC sprite sheets. Different rows have different cycle lengths, so the training loader supports variable-length target sequences. In `train.py`, samples are padded inside a batch, and a `lengths` tensor is used so loss calculations ignore padding frames.

## 3. Transformer Model

The primary model is `SpriteSeq2Seq` in `train.py`. It is a BOS-token encoder-decoder transformer that predicts future sprite frames as sequences of image patches.

### 3.1 Patch Tokenization

Every 64 x 64 RGB frame is divided into non-overlapping 4 x 4 patches. This gives:

- Frame size: 64 x 64.
- Patch size: 4 x 4.
- Patches per frame: 16 x 16 = 256.
- Patch dimension: 4 x 4 x 3 = 48.

The helper function `frame_to_patches` converts a frame into a sequence of 256 patch vectors, and `patches_to_frame` reconstructs full RGB frames from predicted patch vectors. This representation gives the transformer a manageable sequence length while preserving local pixel structure better than treating individual pixels as tokens.

### 3.2 Encoder Context

The encoder receives two forms of information:

- A learned row-label token.
- Patch tokens from the first frame.

The row label is embedded with `nn.Embedding`, producing a single token that represents the desired animation row. This label token is prepended to the encoded first-frame patches. The resulting encoder memory has one label token plus 256 visual tokens.

Each visual token contains three kinds of information:

- Patch color content, projected from 48 dimensions to the transformer width.
- Spatial position, represented by separate learned row and column embeddings.
- Temporal position, represented by sinusoidal frame-position encoding.

For the source frame, the temporal position is frame index 0. This gives the encoder a compact description of what the character looks like and what action type should be generated.

### 3.3 Decoder and BOS Token

The decoder predicts the target sequence using teacher forcing during training. All target frames are patchified and concatenated into one long sequence. A learned beginning-of-sequence token is placed at the front, and the decoder input is shifted by one token so the model predicts the next patch token at each position.

A causal target mask prevents the decoder from seeing future target patches. This makes the training objective match the intended autoregressive structure: future patches and future frames should be inferred from the first frame, the row label, and already-available previous target tokens.

### 3.4 Skip Connection for Sprite Identity

Pixel-art animation depends heavily on preserving identity: palette, clothing, hair, face, weapon shape, and outline must remain consistent across frames. To support this, the transformer adds a patch-level skip connection from the encoded input frame to the decoder output.

The encoded source patches are projected, repeated across the predicted target-frame positions, and added to the decoder activations before the final patch projection. This encourages the model to keep character-specific appearance information available at every generated patch position instead of forcing the decoder to reconstruct all identity details from scratch.

### 3.5 Default Architecture

The default restored transformer configuration is:

| Component | Value |
| --- | --- |
| Model width | 512 |
| Attention heads | 8 |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Feed-forward width | 2048 |
| Dropout | 0.1 |
| Max temporal positions | 64 |
| Row label classes | 21 |
| Input frame channels | RGB |
| Output activation | Sigmoid |

The output projection maps decoder features back to 48-dimensional patch vectors. A sigmoid bounds predicted RGB values to the 0-1 image range.

## 4. Training Pipeline

Training is implemented in `train.py`. The script loads extracted row strips from `data_output/frames`, splits them into an 80/20 train-validation split, and trains the transformer with a combined reconstruction, structure, perceptual, and adversarial objective.

### 4.1 Variable-Length Batches

Different LPC animation rows contain different numbers of frames. The dataset reports the number of frames for each strip, and the collate function pads target sequences to the longest target length in the batch. A mask derived from `lengths` ensures that loss is computed only on real frames.

The optional `FrameCountBucketSampler` groups samples by sequence length. This reduces wasted padding and can improve training efficiency, especially when longer animation rows are mixed with shorter rows.

### 4.2 Curriculum Learning

The training script includes a simple sequence-length curriculum. Early epochs train on short target sequences, starting from `min_frames = 2`. Over the first half of training, the maximum target length ramps toward the longest sequence present in the dataset. This helps the model first learn local motion and appearance preservation before taking on full animation cycles.

### 4.3 Loss Function

The restored transformer does not use pure MSE. The earlier MSE baseline learned rough pose structure but produced blurry pixel-art details because MSE rewards averaged predictions. The current objective is designed to preserve sharper edges and more coherent structure.

The reconstruction loss is:

```text
0.1 * L1 + 0.1 * VGG perceptual + 0.5 * SSIM
```

The L1 term anchors color and local pixel values. The VGG perceptual term compares feature activations rather than raw pixels, which gives the model a higher-level structure signal. SSIM rewards structural similarity and edge alignment, which is especially important for small sprites where a one-pixel error can visibly damage silhouettes.


### 4.4 Patch Discriminator

The project also adds a lightweight patch discriminator. Its role is not to dominate generation, but to nudge predictions toward sharper local texture and cleaner high-frequency detail.

The discriminator is a small convolutional network that maps RGB frames to patch-level logits. It is trained with binary cross-entropy on real target frames and generated frames. Several stabilizers are used:

- GAN warmup: the discriminator begins after 5 epochs.
- Instance noise: Gaussian noise with standard deviation 0.05 is added to real and fake discriminator inputs.
- Real label smoothing: real labels are set to 0.9 instead of 1.0.
- Small adversarial weight: the generator receives only `0.001 * adversarial_loss`.
- Gradient clipping: both generator and discriminator gradients are clipped to 1.0.

This design reflects the main lesson from the project: adversarial loss can improve crispness, but it must remain secondary to reconstruction and structure losses for this small sprite domain.

### 4.5 Optimization

The default training settings are:

| Parameter | Default |
| --- | --- |
| Epochs | 250 |
| Batch size | 128 |
| Learning rate | 1e-4 |
| Minimum learning rate | 1e-6 |
| Generator optimizer | AdamW |
| Generator weight decay | 1e-4 |
| Discriminator optimizer | AdamW |
| Discriminator learning rate | 1e-4 |
| Discriminator betas | 0.5, 0.999 |
| Train-validation split | 80/20 |
| Number of workers | 8 |
| Gradient accumulation | 1 by default, adjusted with curriculum |

The learning-rate schedule uses cosine annealing warm restarts, with the first restart period set to half of the total epoch count. The script supports mixed precision on CUDA, preferring bfloat16 when available and falling back to float16 otherwise. It also supports distributed data parallel training when launched with the appropriate environment variables.



## 5. Comparison to Earlier Approaches

The repository includes earlier or alternative methods, but they mainly serve as context for why the transformer became the focus.

The deterministic warp method is fast and predictable, but it can only move existing pixels. It struggles when an animation requires newly visible limbs, clothing folds, weapons, or body parts. This makes it useful as a simple preview tool, but it has a low quality ceiling.

The MSE transformer baseline learned row-level motion patterns, but its outputs were visibly soft. This is a common failure mode for pixel regression: when multiple plausible futures exist, MSE averages them. In pixel art, that averaging destroys outlines, facial details, small weapons, and palette boundaries.

The restored transformer keeps the useful part of the learned sequence model while changing the objective and architecture details that matter for fidelity. The combination of patch tokens, row conditioning, BOS-token decoding, skip connections, curriculum learning, SSIM, perceptual loss, and a small patch discriminator makes the model better aligned with the actual needs of sprite animation.

Wan2.2 is also represented in the repository, but it is better understood as a high-capacity external comparison rather than the project's main method. General image-to-video models have much more generative power, but they are expensive, less controllable, and trained on natural video priors that do not naturally match 64 x 64 transparent pixel-art sprites.

## 6. Current Strengths

The transformer model has several strengths for the intended domain:

- It is trained directly on LPC-style sprites rather than natural video.
- It conditions on both the source character and the desired animation row.
- It supports variable animation lengths.
- It preserves sprite identity through encoded patch skip connections.
- It produces complete animation strips from one input frame.
- It is much lighter than a large image-to-video model.
- Its outputs are easy to inspect as PNG strips and GIFs.

The model is especially well suited to structured game-art assets where characters share a common sheet layout and animation taxonomy.

## 7. Limitations and Future Work

The current model is still tied to the LPC sprite-sheet domain. It expects 64 x 64 frames, RGB inputs, row labels matching the training layout, and animation structure similar to the generated dataset. New sprite styles, frame sizes, or animation taxonomies would likely require dataset changes and retraining.

The model also generates future frames through the trained decoder path using dummy repeated inputs at inference time. This preserves compatibility with the teacher-forced training architecture, but future work could explore a cleaner fully autoregressive inference loop or scheduled sampling during training.

Possible extensions include:

- Training on more diverse sprite styles and non-LPC assets.
- Adding explicit alpha-channel modeling instead of deriving transparency from near-black pixels.
- Conditioning on natural-language action descriptions in addition to row labels.
- Improving temporal consistency with sequence-level losses.
- Adding palette-aware or edge-aware losses specific to pixel art.
- Building an editor interface where users can preview, select, and revise generated strips.

