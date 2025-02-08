from django.db import models
from django.utils.text import slugify
from django.utils import timezone

# Create your models here.


class Project(models.Model):
    title = models.CharField(max_length=200)
    youtube_url = models.URLField(help_text="Enter the full YouTube URL")
    youtube_id = models.CharField(max_length=20, blank=True, editable=False, default="")
    transcription = models.TextField(blank=True, default="")
    final_video = models.FileField(upload_to="videos/", blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    tag = models.CharField(max_length=50, default="finance")
    is_published = models.BooleanField(default=False)  # New field for published status

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if self.youtube_url and not self.youtube_id:
            # Extract YouTube ID from URL (you might want to implement this logic)
            # For simplicity, let's assume the ID is the last part of the URL
            self.youtube_id = self.youtube_url.split("/")[-1]
        super().save(*args, **kwargs)


class Scene(models.Model):
    MOOD_CHOICES = [
        ("adventure", "Adventure"),
        ("dramatic", "Dramatic"),
        ("happy", "Happy"),
        ("romantic", "Romantic"),
        ("suspense", "Suspense"),
    ]

    project = models.ForeignKey(
        Project, related_name="scenes", on_delete=models.CASCADE
    )
    order = models.PositiveIntegerField(default=0)
    narration = models.TextField()
    image_prompt = models.TextField()
    image = models.ImageField(upload_to='scenes/', null=True, blank=True)
    audio = models.FileField(upload_to='scenes/audio/', null=True, blank=True)
    mood = models.CharField(max_length=20, choices=MOOD_CHOICES, default="happy")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["order"]
        unique_together = ["project", "order"]

    def __str__(self):
        return f"Scene {self.order} of {self.project.title}"

    def save(self, *args, **kwargs):
        if self.order == 0:
            last_scene = (
                Scene.objects.filter(project=self.project).order_by("-order").first()
            )
            self.order = last_scene.order + 1 if last_scene else 1
        super().save(*args, **kwargs)


class YouTubeDetails(models.Model):
    project = models.OneToOneField(
        Project, related_name="youtube_details", on_delete=models.CASCADE
    )
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    thumbnail_title = models.CharField(max_length=100)
    thumbnail_prompt = models.TextField()
    thumbnail_image = models.ImageField(upload_to="thumbnails/", blank=True, null=True)
    thumbnail = models.ImageField(upload_to='thumbnails/', null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"YouTube Details for {self.project.title}"

    def save(self, *args, **kwargs):
        if not self.title:
            self.title = f"Video for {self.project.title}"
        if not self.thumbnail_title:
            self.thumbnail_title = slugify(self.title)[:100]
        super().save(*args, **kwargs)
