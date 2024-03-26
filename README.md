# AI-powered-Language-Preservation
Utilisez des modèles de langage importants pour préserver et revitaliser les langues en voie de disparition en créant des ressources d'apprentissage et des outils de traduction.
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import langid  # Language identification

# Assume 'your_model_name' is the name of a model trained for an endangered language
model_name = 'your_model_name'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Translation pipeline
translation_pipeline = pipeline('translation_xx_to_yy', model=model, tokenizer=tokenizer)

# Language identification function
def identify_language(text):
    lang, _ = langid.classify(text)
    return lang

# Function to generate learning resources (simple translations)
def generate_learning_resources(text, num_examples=5):
    resources = []
    for _ in range(num_examples):
        translated_text = translation_pipeline(text, max_length=40)
        resources.append(translated_text[0]['translation_text'])
    return resources

# Function for translating text
def translate_text(text, target_language='en'):
    if identify_language(text) != target_language:
        return translation_pipeline(text, max_length=40)[0]['translation_text']
    else:
        return "The text is already in the target language."

# Example usage
example_text = "Your example text in the endangered language"
print(f"Identified Language: {identify_language(example_text)}")
print("Learning Resources:", generate_learning_resources(example_text))
print("Translation:", translate_text(example_text))
