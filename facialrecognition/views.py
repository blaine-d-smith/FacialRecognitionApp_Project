from django.shortcuts import render
import os
from django.conf import settings
from .forms import FacialRecognitionForm
from .machine_learning import pipeline_model
from .models import FacialRecognition

# Home
def index(request):
    form = FacialRecognitionForm

    if request.method == 'POST':
        form = FacialRecognitionForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            save = form.save(commit=True)

            # Extract Image
            primary_key = save.pk
            img_obj = FacialRecognition.objects.get(pk=primary_key)
            file_root = str(img_obj.image)
            filepath = os.path.join(settings.MEDIA_ROOT, file_root)
            results = pipeline_model(filepath)
            print(results)

            return render(request, 'pages/index.html', {'form': form, 'upload': True, 'results': results})

    return render(request, 'pages/index.html', {'form': form, 'upload': False})
