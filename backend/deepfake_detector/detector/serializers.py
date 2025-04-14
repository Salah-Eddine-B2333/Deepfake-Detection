from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.CharField(required=True)
    
class VideoUploadSerializer(serializers.Serializer):
    video = serializers.FileField(required=True)