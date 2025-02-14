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
from celery.exceptions import Retry

# Set ImageMagick binary path
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

@shared_task(bind=True, max_retries=3)
def create_narration_task(self, project_id):
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
        try:
            raise self.retry(exc=e, countdown=60)  # Retry after 60 seconds
        except Retry as retry:
            project.narration_status = 'failed'
            project.save()
            raise retry
        except Exception as e:
            project.narration_status = 'failed'
            project.save()
            print(f"Error in create_narration_task: {str(e)}")
            return False

@shared_task
def generate_scene_image_task(scene_id, project_id):
    print(f"[generate_scene_image_task] Received scene_id: {scene_id} (type: {type(scene_id)}), project_id: {project_id}")
    try:
        # Convert scene_id to integer if it's a string
        scene_id = int(scene_id) if isinstance(scene_id, str) else scene_id
        print(f"[generate_scene_image_task] Converted scene_id: {scene_id} (type: {type(scene_id)})")
        scene = Scene.objects.get(id=scene_id)
        print(f"[generate_scene_image_task] Found scene: {scene}")
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
    except Scene.DoesNotExist:
        available_ids = list(Scene.objects.all().values_list('id', flat=True))
        print(f"[generate_scene_image_task] Scene {scene_id} does not exist. Available scene IDs: {available_ids}")
        return False
    except Exception as e:
        try:
            if 'scene' in locals() and scene:
                scene.image_status = 'failed'
                scene.save()
            raise self.retry(exc=e, countdown=60)  # Retry after 60 seconds
        except Retry as retry:
            if 'scene' in locals() and scene:
                scene.image_status = 'failed'
                scene.save()
            raise retry
        except Exception as e:
            if 'scene' in locals() and scene:
                scene.image_status = 'failed'
                scene.save()
            print(f"[generate_scene_image_task] Error: {str(e)}")
            return False

@shared_task
def generate_scene_audio_task(scene_id, project_id):
    print(f"[generate_scene_audio_task] Received scene_id: {scene_id} (type: {type(scene_id)}), project_id: {project_id}")
    try:
        # Convert scene_id to integer if it's a string
        scene_id = int(scene_id) if isinstance(scene_id, str) else scene_id
        print(f"[generate_scene_audio_task] Converted scene_id: {scene_id} (type: {type(scene_id)})")
        scene = Scene.objects.get(id=scene_id)
        print(f"[generate_scene_audio_task] Found scene: {scene}")
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
    except Scene.DoesNotExist:
        available_ids = list(Scene.objects.all().values_list('id', flat=True))
        print(f"[generate_scene_audio_task] Scene {scene_id} does not exist. Available scene IDs: {available_ids}")
        return False
    except Exception as e:
        try:
            if 'scene' in locals() and scene:
                scene.audio_status = 'failed'
                scene.save()
            raise self.retry(exc=e, countdown=60)  # Retry after 60 seconds
        except Retry as retry:
            if 'scene' in locals() and scene:
                scene.audio_status = 'failed'
                scene.save()
            raise retry
        except Exception as e:
            if 'scene' in locals() and scene:
                scene.audio_status = 'failed'
                scene.save()
            print(f"[generate_scene_audio_task] Error: {str(e)}")
            return False

@shared_task(bind=True, max_retries=3)
def generate_video_task(self, project_id, skip_subtitles=False):
    try:
        project = Project.objects.get(id=project_id)
        project_media_dir = os.path.join(settings.MEDIA_ROOT, project.title)
        os.makedirs(project_media_dir, exist_ok=True)
        scenes = project.scenes.all().order_by("order")
        
        if not scenes.exists():
            raise Exception("No scenes found for the project")
        
        width, height = get_image_dimensions(project.video_format)
        video_clips = []
        total_scenes = scenes.count()
        
        for idx, scene in enumerate(scenes, start=1):
            # Update progress for each scene
            self.update_state(state='PROGRESS', meta={'current': idx, 'total': total_scenes, 'message': f'Processing scene {scene.id}'})
            
            if not scene.audio or not scene.image:
                raise Exception(f"Scene {scene.id} missing audio or image")
            
            audio_file_path = os.path.join(settings.MEDIA_ROOT, scene.audio.name)
            if not os.path.exists(audio_file_path):
                raise Exception(f"Audio file not found: {audio_file_path}")
            
            image_file_path = os.path.join(settings.MEDIA_ROOT, scene.image.name)
            if not os.path.exists(image_file_path):
                raise Exception(f"Image file not found: {image_file_path}")
            
            # Create audio and image clips
            audio_clip = AudioFileClip(audio_file_path)
            image_clip = ImageClip(image_file_path).set_duration(audio_clip.duration)
            
            subtitle_clips = []
            if not skip_subtitles:
                transcription_path = os.path.join(project_media_dir, f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_transcription.json")
                transcription = transcribe_audio_with_whisper(audio_file_path, transcription_path)
                
                # Determine subtitle settings
                subtitle_y_position = height - 180 if project.video_format == 'landscape' else int(height * 0.75)
                words_per_group = 4 if project.video_format == 'reel' else 6
                
                for j in range(0, len(transcription), words_per_group):
                    words = transcription[j : j + words_per_group]
                    word_text = " ".join(word_info["word"] for word_info in words)
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
            
            # Create base clip
            base_clip = ColorClip(size=(width, height), color=(0, 0, 0)).set_duration(audio_clip.duration)
            
            # Position image
            image_clip = image_clip.resize(width=width)
            image_y_position = (height - image_clip.h) // 2
            positioned_image = image_clip.set_position(('center', image_y_position))
            
            if project.video_format == 'landscape':
                positioned_image = positioned_image.fx(resize, lambda t: 1 + 0.01 * t)
            
            clips_to_composite = [base_clip, positioned_image]
            if subtitle_clips:
                clips_to_composite.extend(subtitle_clips)
            
            video_clip = CompositeVideoClip(clips_to_composite, size=(width, height)).set_audio(audio_clip)
            video_clips.append(video_clip)
        
        # Update progress before concatenation
        self.update_state(state='PROGRESS', meta={'current': total_scenes, 'total': total_scenes, 'message': 'Concatenating video clips...'})
        final_video = concatenate_videoclips(video_clips, method="compose")
        
        output_video_name = f"final_video_{project.id}.mp4"
        output_video_path = os.path.join(project_media_dir, output_video_name)
        
        # Update progress before writing final video
        self.update_state(state='PROGRESS', meta={'current': total_scenes, 'total': total_scenes, 'message': f'Writing final video to {output_video_path}'})
        final_video.write_videofile(
            output_video_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            threads=4,
            preset="medium"
        )
        
        project.final_video = os.path.join(project.title, output_video_name)
        project.save()
        
        return True
    except Exception as e:
        try:
            raise self.retry(exc=e, countdown=60)  # Retry after 60 seconds
        except Retry as retry:
            raise retry
        except Exception as e:
            print(f"Error in generate_video_task: {str(e)}")
            return False

@shared_task
def test_celery():
    print("Test task is running!")
    return True 