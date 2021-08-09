from django import forms
from .models import FacialRecognition

class FacialRecognitionForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['image'].widget.attrs.update({'class': 'form-control'})

    class Meta:
        model = FacialRecognition
        fields = ['image']
