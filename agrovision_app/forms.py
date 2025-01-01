from django import forms
from .models import ClimateData, CropRecommendation

class ClimateDataForm(forms.ModelForm):
    class Meta:
        model = ClimateData
        fields = ['date', 'precipitation', 'temp_max', 'temp_min', 'wind']

class CropRecommendationForm(forms.ModelForm):
    class Meta:
        model = CropRecommendation
        fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
