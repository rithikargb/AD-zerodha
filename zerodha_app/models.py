from django.db import models

# Create your models here.
class stock_data(models.Model):
    Date = models.DateField()
    AdjClose = models.FloatField()
    Close = models.FloatField()
    High = models.FloatField()
    Low = models.FloatField()
    Open = models.FloatField()
    Volume = models.IntegerField()
    Ticker = models.CharField(max_length=10)