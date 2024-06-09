import fitz
import pytesseract
from PIL import Image
import io
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import chainlit as cl
import torch
from transformers import pipeline, MarianMTModel, MarianTokenizer
from huggingface_hub import HfFolder

# Download NLTK data
nltk.download('punkt')

# Global variables
UPLOAD_FOLDER = 'uploads/'

# Replace with your Hugging Face API token
huggingface_token = "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"
HfFolder.save_token(huggingface_token)

# Initialize models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
classifier_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize translation model for Spanish
translator_model_name = "Helsinki-NLP/opus-mt-en-es"
translator = MarianMTModel.from_pretrained(translator_model_name)
translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
translator.to(device)


def extract_text_from_pdf(pdf_path):
    text = ''
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text.strip()


def extract_text_from_images(pdf_path):
    text = ''
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_text = pytesseract.image_to_string(image)
            text += image_text + ' '
    doc.close()
    return text.strip()


def preprocess_text(input_text):
    clean_text = re.sub(r'[^A-Za-z0-9\s]', '', input_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    sentences = sent_tokenize(clean_text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    return tokenized_sentences


def extract_information(text):
    entities = ner_model(text)
    word_entities = {}
    current_word = ''
    for entity in entities:
        word = entity['word']
        if word.startswith("##"):
            current_word += word[2:]
        else:
            if current_word:
                word_entities[current_word] = entity
            current_word = word
    if current_word:
        word_entities[current_word] = entity
    return word_entities.values()


def classify_document(text):
    if not text:
        return "No text available for classification."
    classes = classifier_model(text, candidate_labels=["finance", "health", "sports", "news", "education"])
    return classes


def translate_text_to_spanish(text):
    input_ids = translator_tokenizer.encode(text, return_tensors="pt").to(device)
    translated_text = translator.generate(input_ids=input_ids)
    return translator_tokenizer.decode(translated_text[0], skip_special_tokens=True)


async def process_pdf(file):
    file_path = file.path
    file_name = file.name

    text_from_pdf = extract_text_from_pdf(file_path)
    text_from_images = extract_text_from_images(file_path)
    text = text_from_pdf + ' ' + text_from_images

    preprocessed_text = preprocess_text(text)
    information = extract_information(text)
    classification = classify_document(text)

    return {
        'filename': file_name,
        'text': text,
        'preprocessed_text': preprocessed_text,
        'information': information,
        'classification': classification
    }


@cl.on_chat_start
async def start():
    cl.user_session.set("pdf_document", None)
    await cl.Message(content="Welcome! Press 1 to upload a PDF file for processing.").send()


@cl.on_message
async def main(message: cl.Message):
    user_document = cl.user_session.get("pdf_document")
    content = message.content.strip().lower()

    if content == "1":
        file = await cl.AskFileMessage(
            content="Please upload a PDF file for processing.",
            accept=["application/pdf"],
            max_size_mb=5,
            max_files=1
        ).send()
        if file and isinstance(file, list) and len(file) > 0:
            document = await process_pdf(file[0])
            cl.user_session.set("pdf_document", document)
            await cl.Message(
                content="PDF File has been processed. To ask for info extracted type 'Info', document classification "
                        "type 'class', to translate to Spanish type 'translate'."
            ).send()
        else:
            await cl.Message(content="No files uploaded. Please upload a PDF file.").send()
    elif user_document:
        if content == "info":
            await display_information_extracted(user_document)
        elif content == "class":
            await display_document_classified(user_document)
        elif content == "translate":
            await translate_document(user_document)
        else:
            await cl.Message(
                content="What would you like to do with the document? To ask for info extracted type 'Info', "
                        "document classification type 'class', to translate to Spanish type 'translate'."
            ).send()
    else:
        await cl.Message(content="Please press 1 to upload a PDF file for processing.").send()


async def display_information_extracted(document):
    text = document['text']
    summary = summarizer(text, max_length=100, min_length=25, do_sample=False)
    summarized_text = summary[0]['summary_text']

    entities = document['information']
    formatted_entities = "\n".join(
        [f"Entity: {entity['word']}, Type: {entity['entity']}, Score: {entity['score']}" for entity in entities])

    await cl.Message(
        content=f"Summarizing key information from the document:\n{summarized_text}\n\nExtracted Entities:\n{formatted_entities}"
    ).send()


async def display_document_classified(document):
    classification = document['classification']
    formatted_classification = "\n".join(
        [f"Label: {label}, Score: {score}" for label, score in zip(classification['labels'], classification['scores'])])
    await cl.Message(
        content=f"Document Classification for {document['filename']}:\n{formatted_classification}"
    ).send()


async def translate_document(document):
    translated_text = translate_text_to_spanish(document['text'])
    await cl.Message(
        content=f"Translated Text of {document['filename']} to Spanish:\n{translated_text}"
    ).send()


if __name__ == "__main__":
    cl.connect()
    cl.listen()
