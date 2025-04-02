from django.urls import path
from .views import DetectDeepfakeView

urlpatterns = [
    path('detect/', DetectDeepfakeView.as_view(), name='detect_deepfake'),
]