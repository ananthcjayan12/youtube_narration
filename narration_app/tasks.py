from celery import shared_task
from .models import Project, Scene, YouTubeDetails
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip, ColorClip, TextClip
from moviepy.video.fx.all import resize
import os
from django.conf import settings
from .views import transcribe_audio_with_whisper, create_word_clip, extract_youtube_id, get_or_create_transcription, create_scenes, create_youtube_details, generate_voice, replicate_run, save_image, resize_with_aspect_ratio
from .views import CreateNarrationView
from django.http import HttpRequest
from django.contrib.auth.models import AnonymousUser
import requests
from django.db.models import Q
from .views import create_scenes_and_youtube_details

@shared_task
def generate_videos_for_all_projects():
    projects = Project.objects.filter(final_video__isnull=True).exclude(final_video='')
    if not projects:
        projects = Project.objects.filter(Q(final_video__isnull=True) | Q(final_video=''))

    for project in projects:
        scenes_ready = all(scene.image and scene.audio for scene in project.scenes.all())

        if scenes_ready:
            generate_video(project)

@shared_task
def process_youtube_url(youtube_url, title):
    print(youtube_url, title)
    youtube_id = extract_youtube_id(youtube_url)
    project, created = Project.objects.get_or_create(
        youtube_id=youtube_id,
        defaults={'title': title, 'youtube_url': youtube_url}
    )
    project_media_dir = os.path.join(settings.MEDIA_ROOT, project.title)
    os.makedirs(project_media_dir, exist_ok=True)
    
    transcription = get_or_create_transcription(youtube_id)
    
    # Create scenes and YouTube details using the logic from CreateNarrationView
    create_scenes_and_youtube_details(project, transcription)
    
    # Generate audio for all scenes
    for scene in project.scenes.all():
        audio_file_name = f"scene_{scene.id}_audio.mp3"
        audio_file_path = os.path.join(project_media_dir, audio_file_name)
        generate_voice(scene.narration, scene.id,project_media_dir)
        if os.path.exists(audio_file_path):
            scene.audio = os.path.join(project.title, audio_file_name)
            scene.save()
        else:
            print(f"Audio file not found: {audio_file_path}")
        scene.save()
    
    # Generate images for all scenes
    for scene in project.scenes.all():
        output = replicate_run(
            "black-forest-labs/flux-schnell",
            input={
                "seed": 5,
                "prompt": scene.image_prompt,
                "num_outputs": 1,
                "aspect_ratio": "16:9",
                "output_format": "png",
                "output_quality": 80,
            },
        )
        image_url = output[0]
        image_data = requests.get(image_url).content
        image_path = os.path.join(settings.MEDIA_ROOT, project.title, f"scene_{scene.id}.png")
        save_image(image_data, image_path)
        resize_with_aspect_ratio(image_path, (1920, 1080))
        scene.image = os.path.join(project.title, f"scene_{scene.id}.png")
        scene.save()
    
    # Generate final video
    generate_video(project)

def generate_video(project):
    project_media_dir = os.path.join(settings.MEDIA_ROOT, project.title)
    os.makedirs(project_media_dir, exist_ok=True)

    video_clips = []
    for scene in project.scenes.all():
        audio_file_path = os.path.join(settings.MEDIA_ROOT, scene.audio.name)
        audio_clip = AudioFileClip(audio_file_path)
        image_file_path = os.path.join(settings.MEDIA_ROOT, scene.image.name)
        image_clip = ImageClip(image_file_path).set_duration(audio_clip.duration)
        
        panned_zoomed_clip = image_clip.fx(
            resize,
            lambda t: 1 + 0.02 * t + (0.01 if t < audio_clip.duration / 2 else -0.01),
        ).set_position(("center", "center"))

        transcription_path = os.path.join(
            project_media_dir,
            f"{os.path.splitext(audio_file_path)[0]}_transcription.json",
        )
        transcription = transcribe_audio_with_whisper(
            audio_file_path, transcription_path
        )

        subtitle_clips = []
        for j in range(0, len(transcription), 2):
            words = transcription[j : j + 2]
            word_text = " ".join([word_info["word"] for word_info in words])
            start_time = words[0]["start"]
            end_time = words[-1]["end"]
            duration = end_time - start_time
            word_clip = create_word_clip(
                word_text,
                start_time,
                duration,
                panned_zoomed_clip.w,
                panned_zoomed_clip.h,
            )
            word_clip = word_clip.set_position(
                ("center", panned_zoomed_clip.h - 180)
            )
            subtitle_clips.append(word_clip)

        video_clip = CompositeVideoClip(
            [panned_zoomed_clip, *subtitle_clips]
        ).set_audio(audio_clip)
        video_clips.append(video_clip)

    final_video = concatenate_videoclips(video_clips, method="compose")
    output_video_name = f"final_video_{project.id}.mp4"
    output_video_path = os.path.join(project_media_dir, output_video_name)
    final_video.write_videofile(output_video_path, codec="libx264", fps=10)

    project.final_video = os.path.join(project.title, output_video_name)
    project.save()
