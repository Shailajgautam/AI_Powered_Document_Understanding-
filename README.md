# AI Powered Document Understanding

## Overview

This project involves creating a chatbot capable of processing PDF files. The chatbot can extract text, perform information extraction, classify the document, and translate the text into Spanish. The key functionalities are built using several advanced libraries and models, including NLP models for summarization, named entity recognition (NER), zero-shot classification, and translation.

## Approach

### Libraries Used

1. **PyMuPDF (fitz)**: Used for extracting text and images from PDF files.
2. **pytesseract**: An Optical Character Recognition (OCR) tool used to extract text from images.
3. **PIL (Pillow)**: Used to handle image data.
4. **re (Regular Expressions)**: Used for text preprocessing.
5. **nltk (Natural Language Toolkit)**: Used for text tokenization.
6. **chainlit**: A framework for building chatbots.
7. **torch**: PyTorch, used for handling deep learning models.
8. **transformers (Hugging Face)**: Provides pre-trained models for various NLP tasks.
9. **huggingface_hub**: Used to manage Hugging Face API tokens.

### Preprocessing Steps

1. **Text Cleaning**: Removal of non-alphanumeric characters and extra whitespace.
2. **Sentence Tokenization**: Splitting text into sentences.
3. **Word Tokenization**: Splitting sentences into words.

### OCR Engine Selection

The project uses Tesseract (pytesseract), a widely used OCR engine known for its accuracy and ease of integration with Python.

## Large Language Model (LLM)

### Chosen LLM

The project leverages several models from the Hugging Face library:

1. **Summarization**: `sshleifer/distilbart-cnn-12-6`
2. **NER**: `dbmdz/bert-large-cased-finetuned-conll03-english`
3. **Zero-Shot Classification**: `facebook/bart-large-mnli`
4. **Translation**: `Helsinki-NLP/opus-mt-en-es`

### Integration with the Pipeline

1. **Summarization**: The summarizer model condenses the text extracted from the PDF into a concise summary.
2. **Information Extraction**: The NER model identifies and extracts named entities from the text.
3. **Classification**: The zero-shot classification model determines the category of the document based on its content.
4. **Translation**: The MarianMT model translates the text from English to Spanish.

### Techniques Employed

- **Information Extraction**: Using NER to identify key entities in the text.
- **Classification**: Using zero-shot learning to classify documents into predefined categories without needing training data for each category.
- **Translation**: Using a pre-trained MarianMT model for translation.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shailajgautam/AI_Powered_Document_Understanding-
   cd AI_Powered_Document_Understanding-
   ```

2. Create a virtual environment and activate it: (Optional)

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Hugging Face API token in .env file:


### Running the Chatbot

1. Run the main script:
   ```bash
   chainlit run app.py -w
   ```

2. Follow the on-screen instructions to interact with the chatbot.

## Interaction Instructions

- **Upload a PDF**: Start by pressing `1` to upload a PDF file for processing.
- **Request Information**: After uploading, you can request:
  - Information extraction by typing `Info`.
  - Document classification by typing `class`.
  - Translation to Spanish by typing `translate`.

## Design Choices and Technologies

- **PyMuPDF and pytesseract**: Chosen for their efficiency and ease of use in extracting text and images from PDFs.
- **Hugging Face Transformers**: Utilized for state-of-the-art NLP models, providing robust solutions for summarization, NER, classification, and translation.
- **Chainlit**: Selected for building an interactive chatbot interface.

This setup provides a powerful and flexible system for processing PDF documents, extracting meaningful information, and offering additional functionalities like classification and translation, making it a comprehensive tool for document analysis and management.
