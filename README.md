# Paligemma - From Scratch In PyTorch

## Abstract
[PaliGemma](https://arxiv.org/pdf/2407.07726) is an open Vision-Language Model (VLM) that is based on the SigLIP-So400m vision encoder and the Gemma-2B language model. It is trained to be a versatile and broadly knowledgeable base model that is effective to transfer. It achieves strong performance on a wide variety of open-world tasks. We evaluate PaliGemma on almost 40 diverse tasks including standard VLM benchmarks, but also more specialized tasks such as remote-sensing and segmentation.

<b>Caution</b>: The number of parameters in this implementation might not match the actual model, as some modifications have been made due to resource constraints.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/paligemma.git
   cd paligemma
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
3. Training
   ```sh
   python train_model.py
## Project Structure

```sh
PaliGemma/
    │── gemma.py          # Language model integration
    │── paligemma.py      # Main model script
    │── processor.py      # Preprocessing module
    │── siglip.py         # Vision encoder module
    │── train_model.py    # Training script
    │── requirements.txt  # Dependencies
