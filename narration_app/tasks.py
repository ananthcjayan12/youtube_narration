from celery import shared_task
from .models import Project, Scene, YouTubeDetails
from django.conf import settings
import os
from moviepy.editor import (
    ImageClip,
    concatenate_videoclips,
    AudioFileClip,
    ColorClip,
    TextClip,
    CompositeVideoClip,
)
from moviepy.video.fx.all import resize
import replicate
import requests
from PIL import Image
import io
from django.core.files.base import ContentFile
from .utils import (
    get_or_create_transcription,
    create_scenes_and_youtube_details,
    get_image_dimensions,
    create_word_clip,
    transcribe_audio_with_whisper,
    generate_voice,
)
from moviepy.config import change_settings

# Set ImageMagick binary path
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

@shared_task
def create_narration_task(project_id):
    try:
        project = Project.objects.get(id=project_id)
        project.narration_status = 'processing'
        project.save()

        youtube_id = project.youtube_id
        transcription = get_or_create_transcription(youtube_id)
        create_scenes_and_youtube_details(project, transcription)

        project.narration_status = 'completed'
        project.save()
        return True
    except Exception as e:
        project.narration_status = 'failed'
        project.save()
        print(f"Error in create_narration_task: {str(e)}")
        return False

@shared_task
def generate_scene_image_task(scene_id, project_id):
    try:
        scene = Scene.objects.get(id=scene_id)
        scene.image_status = 'processing'
        scene.save()

        project = Project.objects.get(id=project_id)
        project_media_dir = os.path.join(settings.MEDIA_ROOT, project.title)
        os.makedirs(project_media_dir, exist_ok=True)

        width, height = get_image_dimensions(project.video_format)
        aspect_ratio = "9:16" if project.video_format == 'reel' else "16:9"

        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "seed": 5,
                "prompt": scene.image_prompt,
                "num_outputs": 1,
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
                "output_quality": 80,
            },
        )
        image_url = output[0]
        image_data = requests.get(image_url).content
        image_path = os.path.join(project_media_dir, f"scene_{scene.id}.png")
        
        # Save and resize image
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        image = Image.open(image_path)
        image = image.resize((width, height), Image.LANCZOS)
        image.save(image_path)
        
        scene.image = os.path.join(project.title, f"scene_{scene.id}.png")
        scene.image_status = 'completed'
        scene.save()
        
        return True
    except Exception as e:
        scene.image_status = 'failed'
        scene.save()
        print(f"Error in generate_scene_image_task: {str(e)}")
        return False

@shared_task
def generate_scene_audio_task(scene_id, project_id):
    try:
        scene = Scene.objects.get(id=scene_id)
        scene.audio_status = 'processing'
        scene.save()

        project = Project.objects.get(id=project_id)
        project_media_dir = os.path.join(settings.MEDIA_ROOT, project.title)
        os.makedirs(project_media_dir, exist_ok=True)
        
        audio_file_name = f"scene_{scene.id}_audio.mp3"
        audio_file_path = os.path.join(project_media_dir, audio_file_name)
        
        # Generate the audio file
        generate_voice(scene.narration, scene.id, project_media_dir)
        
        if os.path.exists(audio_file_path):
            scene.audio = os.path.join(project.title, audio_file_name)
            scene.audio_status = 'completed'
            scene.save()
            return True

        scene.audio_status = 'failed'
        scene.save()
        return False
    except Exception as e:
        scene.audio_status = 'failed'
        scene.save()
        print(f"Error in generate_scene_audio_task: {str(e)}")
        return False

@shared_task
def test_celery():
    print("Test task is running!")
    return True

@shared_task
def generate_video_task(project_id):
    print(f"Starting video generation for project {project_id}")
    try:
        project = Project.objects.get(id=project_id)
        project_media_dir = os.path.join(settings.MEDIA_ROOT, project.title)
        print(f"Project media directory: {project_media_dir}")
        os.makedirs(project_media_dir, exist_ok=True)
        scenes = project.scenes.all().order_by("order")
        
        if not scenes.exists():
            print("No scenes found for the project")
            return False

        width, height = get_image_dimensions(project.video_format)
        print(f"Video dimensions: {width}x{height}")
        video_clips = []

        for scene in scenes:
            print(f"Processing scene {scene.id}")
            
            # Check if audio and image exist
            if not scene.audio or not scene.image:
                print(f"Scene {scene.id} missing audio or image")
                continue

            audio_file_path = os.path.join(settings.MEDIA_ROOT, scene.audio.name)
            if not os.path.exists(audio_file_path):
                print(f"Audio file not found: {audio_file_path}")
                continue

            image_file_path = os.path.join(settings.MEDIA_ROOT, scene.image.name)
            if not os.path.exists(image_file_path):
                print(f"Image file not found: {image_file_path}")
                continue

            try:
                audio_clip = AudioFileClip(audio_file_path)
                print(f"Audio duration: {audio_clip.duration}")
                
                image_clip = ImageClip(image_file_path).set_duration(audio_clip.duration)
                print(f"Created image clip for scene {scene.id}")
                
                transcription_path = os.path.join(
                    project_media_dir,
                    f"{os.path.splitext(audio_file_path)[0]}_transcription.json",
                )
                transcription = transcribe_audio_with_whisper(audio_file_path, transcription_path)
                print(f"Transcription completed for scene {scene.id}")

                # Create base clip
                base_clip = ColorClip(size=(width, height), color=(0, 0, 0))
                base_clip = base_clip.set_duration(audio_clip.duration)

                # Position image
                image_clip = image_clip.resize(width=width)
                image_y_position = (height - image_clip.h) // 2
                positioned_image = image_clip.set_position(('center', image_y_position))

                if project.video_format == 'landscape':
                    positioned_image = positioned_image.fx(
                        resize,
                        lambda t: 1 + 0.01 * t,
                    )

                # Calculate subtitle position and settings
                subtitle_y_position = height - 180 if project.video_format == 'landscape' else int(height * 0.75)
                words_per_group = 4 if project.video_format == 'reel' else 6

                subtitle_clips = []
                for j in range(0, len(transcription), words_per_group):
                    words = transcription[j : j + words_per_group]
                    word_text = " ".join([word_info["word"] for word_info in words])
                    start_time = words[0]["start"]
                    end_time = words[-1]["end"]
                    duration = end_time - start_time

                    word_clip = create_word_clip(
                        word_text,
                        start_time,
                        duration,
                        width,
                        height,
                        font_size=80 if project.video_format == 'reel' else 50
                    )

                    if word_clip is not None:
                        word_clip = word_clip.set_position(('center', subtitle_y_position))
                        subtitle_clips.append(word_clip)

                # Combine clips
                clips_to_composite = [base_clip, positioned_image]
                if subtitle_clips:
                    clips_to_composite.extend(subtitle_clips)

                video_clip = (CompositeVideoClip(
                    clips_to_composite,
                    size=(width, height)
                ).set_audio(audio_clip))

                video_clips.append(video_clip)
                print(f"Scene {scene.id} processed successfully")

            except Exception as scene_error:
                print(f"Error processing scene {scene.id}: {str(scene_error)}")
                continue

        if not video_clips:
            print("No video clips were created")
            return False

        print("Concatenating video clips...")
        final_video = concatenate_videoclips(video_clips, method="compose")
        output_video_name = f"final_video_{project.id}.mp4"
        output_video_path = os.path.join(project_media_dir, output_video_name)
        
        print(f"Writing final video to {output_video_path}")
        final_video.write_videofile(
            output_video_path,
            codec="libx264",
            fps=10
        )

        project.final_video = os.path.join(project.title, output_video_name)
        project.save()
        print("Video generation completed successfully")
        
        return True
    except Exception as e:
        print(f"Error in generate_video_task: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False 