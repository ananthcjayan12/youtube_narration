from django import forms
from .models import Project, Scene, YouTubeDetails


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ["title", "youtube_url", "video_format"]


class SceneForm(forms.ModelForm):
    class Meta:
        model = Scene
        fields = ["narration", "image_prompt"]


class YouTubeDetailsForm(forms.ModelForm):
    class Meta:
        model = YouTubeDetails
        fields = ["title", "description", "thumbnail_title", "thumbnail_prompt"]


class CustomScriptForm(forms.Form):
    custom_script = forms.CharField(widget=forms.Textarea, label="Enter your custom script or idea")
