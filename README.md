## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Entity Recognition: Identify and classify named entities (e.g., persons, organizations, locations) from unstructured text using a pre-trained and fine-tuned model.
### DESIGN STEPS:

#### STEP 1:Model Selection and Fine-Tuning
Choose a pre-trained BART model (e.g., facebook/bart-large). Fine-tune the model on a NER dataset (e.g., CoNLL-03

#### STEP 2:Pipeline Creation
Load the fine-tuned BART model and tokenizer using the transformers library. Create a NER pipeline to extract named entities from input text.

#### STEP 3:Gradio Interface Development
Develop a Gradio interface to accept text input and display extracted entities. Deploy the interface for user interaction.

### PROGRAM:
```
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import gradio as gr

# Load pre-trained BERT NER model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a pipeline for NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Function to process user input
def ner_function(text):
    entities = ner_pipeline(text)
    return "\n".join([f"{ent['word']} ({ent['entity']})" for ent in entities])

# Gradio Interface
iface = gr.Interface(
    fn=ner_function,
    inputs=gr.Textbox(lines=5, label="Input Text"),
    outputs=gr.Textbox(lines=10, label="Named Entities"),
    title="NER Demo with Pre-trained Model"
)

iface.launch()
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/8cb52432-4593-43ee-84e8-cd5abf868f09)

### RESULT:
A fine-tuned BART model successfully identifies and classifies named entities in text.The NER pipeline efficiently extracts entities such as persons, organizations, and locations.The Gradio interface provides an intuitive platform for users to input text and view the results interactively.overall,the Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework implemented successfully
