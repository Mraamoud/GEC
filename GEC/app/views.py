from django.shortcuts import render
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
from django.conf import settings
import os
model_path_bart = os.path.join(settings.BASE_DIR, "app", "bart-autocorrector-final")
model_path_t5 = os.path.join(settings.BASE_DIR, "app", "final-model")

model_bart = BartForConditionalGeneration.from_pretrained(model_path_bart)
tokenizer_bart = BartTokenizer.from_pretrained(model_path_bart)

model_t5 = T5ForConditionalGeneration.from_pretrained(model_path_t5)
tokenizer_t5 = T5Tokenizer.from_pretrained(model_path_t5)

model_bart.eval()
model_bart.to("cuda" if torch.cuda.is_available() else "cpu")

def autocorrect_bart(text):
    inputs = tokenizer_bart([text], return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model_bart.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_bart.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)

    return tokenizer_bart.decode(outputs[0], skip_special_tokens=True)


model_t5.eval()
model_t5.to("cuda" if torch.cuda.is_available() else "cpu")
def autocorrect_t5(text):
    inputs = tokenizer_t5([text], return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model_t5.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_t5.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)

    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True)

# Create your views here.
def home_view(request) : 
    if request.method == "POST" : 
        elected_model = request.POST.get("model")
        prompt_text = request.POST.get("prompt") 
        corrected_sentence = ""
        if elected_model == "Bart" : 
            corrected_sentence = autocorrect_bart(prompt_text)
            for i in range(10) :
                corrected_sentence = autocorrect_bart(corrected_sentence)
        elif elected_model == "T5" : 
            corrected_sentence = autocorrect_t5(prompt_text)
            for i in range(10) :
                corrected_sentence = autocorrect_t5(corrected_sentence)
        context = {
            "prompt": prompt_text,
            "correction": corrected_sentence,
        }
        return render(request, "app/index.html", context)
    return render(request=request, template_name="app/index.html",)
