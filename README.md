# **DeepSeek-OCR-2-Demo**

A Gradio-based interactive web application for **DeepSeek-OCR-2**, a multimodal model designed for advanced optical character recognition and document understanding. This application allows users to perform various OCR tasks such as converting documents to markdown, extracting text, locating specific text within images, and parsing figures, all through a user-friendly interface. This demo leverages the `deepseek-ai/DeepSeek-OCR-2` model with `flash-attention-2` for efficient inference on NVIDIA GPUs.

## Features

* **Multiple Task Modes**:
* **Markdown**: Converts document images directly into structured markdown.
* **Free OCR**: General purpose text extraction.
* **OCR Image**: Performs OCR with grounding capabilities.
* **Parse Figure**: Specifically optimized for understanding and describing figures or charts.
* **Locate**: Finds specific text within an image and highlights it with bounding boxes.
* **Describe**: Provides a detailed textual description of the image content.
* **Custom**: Allows users to input custom prompts for specialized tasks.


* **Visual Grounding**: Supports bounding box visualization for located text and detected elements.
* **Resolution Control**: Offers multiple processing resolutions (Default, Quality, Fast, No Crop, Small) to balance speed and accuracy.
* **Markdown Preview**: Renders extracted text as markdown for easy verification.
* **Cropped Element Extraction**: Automatically extracts and displays cropped images of detected figures or regions of interest.

<img width="1647" height="852" alt="Screenshot 2026-01-28 074204" src="https://github.com/user-attachments/assets/ffad4a68-2a78-4ca7-a605-22b3f7adb5fc" />
<img width="1618" height="856" alt="Screenshot 2026-01-28 074226" src="https://github.com/user-attachments/assets/135396bd-19ad-4bd1-8acd-0079acd654b5" />

## Prerequisites

* Python 3.10
* NVIDIA GPU with CUDA support (Application uses `torch.bfloat16` and `flash-attention-2`).
* Linux environment (recommended for Flash Attention compatibility).

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/PRITHIVSAKTHIUR/DeepSeek-OCR-2-Demo.git
cd DeepSeek-OCR-2-Demo

```


2. **Install Dependencies**
It is recommended to use a virtual environment. Install the required packages using `pip`.
*Note: The `requirements.txt` includes a direct link to a precompiled Flash Attention wheel. Ensure your CUDA and Torch versions match if you are not using the specific setup listed below.*
```bash
pip install -r requirements.txt

```


**requirements.txt content:**
```text
flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
transformers==4.46.3
tokenizers==0.20.3
torch==2.6.0
torchvision
easydict
einops
addict
gradio

```



## Usage

1. **Run the Application**
Execute the main python script to launch the Gradio server.
```bash
python deepseek_ocr_v2_demo.py

```


2. **Access the Interface**
Open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).
3. **Perform OCR**
* **Upload Image**: Drag and drop or select an image (JPG, PNG, etc.).
* **Select Resolution**: Choose a mode (e.g., "Default" for general use, "Quality" for dense text).
* **Select Task**: Choose the specific operation you want to perform (e.g., "Markdown", "Locate").
* **Prompt (Optional)**: If using "Custom" or "Locate" modes, enter your text query.
* **Click "Perform OCR"**: The model will process the image and display results in the tabs on the right.



## Application Structure

* `deepseek_ocr_v2_demo.py`: The main entry point containing the Gradio UI logic and model inference pipeline.
* `requirements.txt`: Python dependencies.
* `examples/`: Directory containing sample images for testing (ensure this folder exists if running locally with examples).

## Technical Details

* **Model**: `deepseek-ai/DeepSeek-OCR-2`
* **Precision**: `bfloat16`
* **Attention Mechanism**: `flash_attention_2`
* **Framework**: PyTorch 2.6.0, Transformers 4.46.3, Gradio

## License

Please refer to the repository for license information regarding the code and the model usage.

## Repository

[https://github.com/PRITHIVSAKTHIUR/DeepSeek-OCR-2-Demo.git](https://github.com/PRITHIVSAKTHIUR/DeepSeek-OCR-2-Demo.git)
