#IOG Detection using LMM - User Guide

1) Overview

This framework tests multiple LLM models including GPT, Claude, and Gemini on image classification tasks and provides ensemble analysis capabilities. The system processes images sequentially through each model and generates comparative results for ensemble method evaluation.

2) Configuration

The framework is currently configured to use GPT-4o, Claude-3-5-Sonnet-20241022, and Gemini-2.5-Flash-Preview models. Users can modify these settings by updating the model configuration section in the main Python file. Before running the program, users must add their own API keys to the .env file as the distributed version will not contain pre-configured keys for security reasons. API keys must be kept confidential at all times once entered. Test images should be placed in the designated img/ directory before execution.

3) Program Workflow

The program begins by prompting users to specify the starting image number, with the default being image 1 if no input is provided. Images are then processed sequentially through GPT, Claude, and Gemini models in batches of 10 images. All results are automatically saved as timestamped CSV files in the main directory, while execution logs are stored in the logs/ folder for debugging purposes.

4) Installation and Execution

Users must first install the required dependencies using pip install -r requirements.txt or pip3 install -r requirements.txt depending on their Python installation. Ensure that valid API keys for OpenAI, Anthropic, and Google are properly configured in the .env file before attempting to run the program. Once dependencies are installed and API keys are configured, the framework can be executed using python test.py or python3 test.py commands.

5) Critical Usage Notes

The framework includes file list information that is automatically appended to prompts within each model function to ensure proper file recognition. While some prompt inconsistencies may exist due to iterative development, users can adjust these as needed. For cost management purposes, users conducting prompt testing should terminate execution after 10 images using Ctrl+C to avoid excessive API charges, as processing the complete dataset of 1600+ images will incur significant costs.

6) Technical Limitations and Support

Claude Opus model integration currently experiences compatibility issues that prevent proper result parsing, though other models function correctly. Users should monitor API usage and credit consumption carefully during experiments.
