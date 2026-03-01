# web.py
# Minimal Django skeleton for the web interface

import os
from django.shortcuts import render
from django.http import JsonResponse
from transcribe import transcribe_audio
from score import evaluate_pronunciation
from question import get_question

# TODO: Setup Django urls, views, and templates

def home(request):
    """
    Home page view: display welcome and instructions
    """
    # TODO: render template
    return render(request, "home.html")

def get_question(request):
    """
    API endpoint to provide a random question
    """
    question = get_question()
    return JsonResponse({"question": question})

def submit_audio(request):
    """
    API endpoint to receive user audio, transcribe, and return pronunciation score
    """
    # TODO: handle uploaded audio
    transcript = ""
    score_report = {}
    return JsonResponse({"transcript": transcript, "score": score_report})