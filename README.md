# 2d Animation Generation From Single Sprites


## Introduction

### Main Goal:
This project will be built around the Python-Rust based 2d GameEngine I'm working on. The goal is to add a animation generation system directly into the Game Editor, allowing developers to create common animations (Running, jumping, attacking .etc) from an idle frame, without drawing each individual frames.



## The Two Approaches

### The Baseline Warp Approach (Thin Plate Spline)

#### Big Idea:

The first solution is simpler and involves defining a set of "joints" and creating a rig. A motion library is also defined for certain actions, holding the position of the joints in each frame. For each frame, Thin Plate Spline Warp is applied to deform the source sprite's pixels to match the target joint positions, using an RBF solver with TPS kernel: φ(r) = r²·log(r). This kernel minimises the bending energy and makes sure that the warp deforms as little as possible around the joints. 


#### Measuring Smoothness Around the Joints

The Think plate spline considers the integral of the square of the second derivative. the TPS fits a mapping function to minimize the following energy function:

$E_{tps}(f) = \sum_{i=1}^{k}||y_i - f(x_i)||^2$

#### Limitations:

Due to difference in artstyles and character propotions used by different artists, it is not ideal to assume character propotions. This leaves us wtih two options: joint identification through a framework like MediaPipe and assign joints to sprite, or manually defining a set of joints on the first frame. 

After extensive testing with MediaPipe, it can be concluded that the model is built for realistic images with human propotions, while most pixel-style 2d assets have unrealistic propotions, making it extremely challenging for the model to identify the correct pose. As a result, I've opted to set up a function in the Game Editor UI interface to manually define the joints on the sprite before generating animations, making the movements more realistic. 



### The GenAI Approach

For the generative approach, I have yet to settle on a final method, but here's a candidate worth exploring.

#### Image to Sprite Sheet Through Finetuning Stable Diffusion

The idea here is to finetune a latent diffusion model to map a single idle frame to a full animation sprite sheet. **Stable Diffusion** is a well-established open-source latent diffusion model with publicly available weights and a rich ecosystem of finetuning techniques (LoRA, DreamBooth, ControlNet, etc.), making it a practical starting point. The goal would be to condition the model on the input idle frame and have it generate a structurally correct sprite sheet — same character, consistent palette and proportions, across an action sequence laid out in the standard grid format.

Finetuning approaches like LoRA are particularly appealing here since they're relatively lightweight and can adapt the model to a specific domain (pixel-art sprite sheets) without requiring massive compute. A ControlNet-style setup could also work well, where the idle frame acts as the conditioning input and the model learns to "fill in" the remaining animation frames.

#### Weaknesses of This Approach

- **Training data is limited.** The LPC dataset is clean and structured, but it's a single art style. A model finetuned only on LPC will likely struggle to generalise to characters with very different proportions or aesthetics.
- **Identity drift across frames.** Getting the model to produce a character that's clearly the same person in frame 1 and frame 8 is harder than it sounds — diffusion models don't have an explicit notion of "this is the same entity." This is probably the biggest challenge.
- **Grid layout coherence.** Generating a well-structured sprite sheet (correct row/column alignment, consistent frame sizing) is a fairly specific output format that the base model knows nothing about — it all has to be learned from the finetuning data.
- **Evaluation is tricky.** Metrics like SSIM and LPIPS measure pixel similarity to ground truth, but a generated frame that looks visually correct may still score poorly if it doesn't match the reference pixel-for-pixel. Worth keeping that in mind when interpreting results.

#### Candidate Models for Sequence Image Generation

A few specific models worth considering for this task:

**Stable Video Diffusion (SVD)** — Stability AI's image-to-video model. Takes a single image and generates a short, temporally consistent sequence from it, which maps directly onto the "idle frame → animation frames" problem.
- Pro: Designed exactly for image-conditioned sequence generation; identity preservation is built into the architecture
- Con: Produces continuous smooth motion rather than discrete pose frames; not trained on pixel art, so the output style may drift


**CogVideoX** — An open-source video generation model from Zhipu AI, built on a diffusion transformer backbone with strong image-to-video support.
- Strong open-source video generation with good temporal coherence and image conditioning
- Much heavier than the SD-based options; harder to finetune on a small domain-specific dataset like LPC


## Datasets

### Universal LPC Spritesheet

The primary dataset is the **Universal LPC (Liberated Pixel Cup) Spritesheet**, a community-maintained collection of modular human character sprites in a standardised grid layout. Each sheet follows a fixed format: rows correspond to animation types (walk, run, slash, hurt, etc.) and columns correspond to frames within each cycle. All characters share the same pixel dimensions and joint proportions, making it ideal for supervised learning where the input-output mapping is structurally consistent.

The dataset is particularly well suited for this task because every sheet already pairs a static idle frame (the first frame of the walk-down row) with the full animation grid, providing a natural supervised signal for the image-to-sprite-sheet task without any additional annotation.

### OpenGameArt.org

A broader collection of community-contributed 2d game assets released under open licenses. Unlike the LPC dataset, OpenGameArt spans a wide range of art styles, character proportions, and animation conventions, introducing the style variation needed to prevent the model from overfitting to LPC's specific aesthetic. Sprite sheets are filtered to include only those with clearly segmented animation rows and consistent frame grids.


## Evaluation Plan

#### Dataset Split

Evaluation uses the Universal LPC Spritesheet dataset exclusively, as its standardised grid format provides reliable ground-truth frame pairs. The component layers are composited into full character sheets and then split as follows:

Train 90%, Test 10%


#### Metrics

**Visual Fidelity** — SSIM and LPIPS are computed between each generated frame and the corresponding ground-truth frame in the test set. These measure pixel-level sharpness and similarity.

**Identity Preservation** — CLIP cosine similarity between the input idle frame and each generated frame checks that the diffusion model has preserved the character's visual style, palette, and proportions across the animation sequence rather than regressing to an average LPC appearance.


