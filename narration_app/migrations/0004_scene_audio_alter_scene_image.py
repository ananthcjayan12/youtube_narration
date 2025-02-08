# Generated by Django 4.2.15 on 2024-09-10 11:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("narration_app", "0003_alter_scene_options_project_updated_at_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="scene",
            name="audio",
            field=models.FileField(blank=True, null=True, upload_to="scenes/audio/"),
        ),
        migrations.AlterField(
            model_name="scene",
            name="image",
            field=models.ImageField(blank=True, null=True, upload_to="scenes/"),
        ),
    ]
