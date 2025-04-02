from django.db import models

class Detection(models.Model):
    image = models.ImageField(upload_to='uploads/')
    timestamp = models.DateTimeField(auto_now_add=True)
    result = models.JSONField(null=True)