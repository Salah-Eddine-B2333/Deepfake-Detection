from django.urls import path
from .views import DetectDeepfakeView, VideoDeepfakeDetectionView

urlpatterns = [
    path('detect-image/', DetectDeepfakeView.as_view(), name='detect_deepfake'),
    path('detect-video/', VideoDeepfakeDetectionView.as_view(), name='detect_video_deepfake'),
]