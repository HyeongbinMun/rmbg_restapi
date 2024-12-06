# Generated by Django 4.0.3 on 2024-09-04 05:23

import WebAnalyzer.utils.filename
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImageModel',
            fields=[
                ('image', models.ImageField(upload_to=WebAnalyzer.utils.filename.default)),
                ('token', models.AutoField(primary_key=True, serialize=False)),
                ('uploaded_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(auto_now=True)),
                ('result_image', models.ImageField(upload_to=WebAnalyzer.utils.filename.get_upload_to)),
                ('result', models.JSONField(null=True)),
            ],
        ),
    ]
