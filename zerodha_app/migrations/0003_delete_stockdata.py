# Generated by Django 5.1.1 on 2025-01-10 17:38

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('zerodha_app', '0002_stockdata_delete_stockdatatable'),
    ]

    operations = [
        migrations.DeleteModel(
            name='stockData',
        ),
    ]
