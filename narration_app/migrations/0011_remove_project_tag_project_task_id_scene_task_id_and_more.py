# Generated by Django 4.2.15 on 2025-02-14 06:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("narration_app", "0010_remove_project_narration_task_id_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="project",
            name="tag",
        ),
        migrations.AddField(
            model_name="project",
            name="task_id",
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name="scene",
            name="task_id",
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name="project",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True),
        ),
        migrations.AlterField(
            model_name="project",
            name="transcription",
            field=models.TextField(blank=True),
        ),
        migrations.AlterField(
            model_name="project",
            name="video_format",
            field=models.CharField(
                choices=[
                    ("reel", "Instagram Reel (9:16)"),
                    ("landscape", "Landscape (16:9)"),
                ],
                default="reel",
                max_length=20,
            ),
        ),
        migrations.AlterField(
            model_name="project",
            name="youtube_id",
            field=models.CharField(blank=True, max_length=20),
        ),
        migrations.AlterField(
            model_name="project",
            name="youtube_url",
            field=models.URLField(),
        ),
        migrations.AlterField(
            model_name="scene",
            name="audio",
            field=models.FileField(blank=True, null=True, upload_to="audio/"),
        ),
        migrations.AlterField(
            model_name="scene",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True),
        ),
        migrations.AlterField(
            model_name="scene",
            name="image_prompt",
            field=models.TextField(blank=True),
        ),
        migrations.AlterField(
            model_name="scene",
            name="order",
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name="youtubedetails",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True),
        ),
        migrations.AlterField(
            model_name="youtubedetails",
            name="thumbnail_prompt",
            field=models.TextField(blank=True),
        ),
        migrations.AlterField(
            model_name="youtubedetails",
            name="thumbnail_title",
            field=models.CharField(blank=True, max_length=200),
        ),
    ]
