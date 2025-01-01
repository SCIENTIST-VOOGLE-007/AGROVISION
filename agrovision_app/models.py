from django.db import models

class ClimateData(models.Model):
    date = models.DateField()
    precipitation = models.FloatField()
    temp_max = models.FloatField()
    temp_min = models.FloatField()
    wind = models.FloatField()
    climate = models.CharField(max_length=100)

class CropRecommendation(models.Model):
    N = models.FloatField()
    P = models.FloatField()
    K = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()
    rainfall = models.FloatField()
    label = models.CharField(max_length=100)
