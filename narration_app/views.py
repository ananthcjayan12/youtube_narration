from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from .models import Project, Scene, YouTubeDetails
from .forms import ProjectForm, SceneForm, YouTubeDetailsForm
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import os
from dotenv import load_dotenv
from moviepy.editor import (
    ImageClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeAudioClip,
    ColorClip,
    TextClip,
    CompositeVideoClip,
)
from moviepy.video.fx.all import resize
from tenacity import retry, wait_random_exponential, stop_after_attempt
from replicate import run as replicate_run
import json
import requests
from django.conf import settings
from django.core.files import File
from PIL import Image
from django.urls import reverse
from django.views.generic.edit import FormView
from .forms import CustomScriptForm  # You'll need to create this form
from django.http import JsonResponse
from django.shortcuts import redirect
from django.views import View
from moviepy.editor import concatenate_videoclips
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import io
import csv
from io import StringIO
from celery import chain
from PIL import Image, ImageDraw, ImageFont
from django.db.models import Q  # Ensure to import Q for filtering
import csv
from django.http import HttpResponse
from requests.auth import HTTPProxyAuth
from urllib.parse import urlparse
import random
from .tasks import (
    create_narration_task,
    generate_scene_image_task,
    generate_scene_audio_task,
    generate_video_task,
)

class GenerateCSVView(View):
    def get(self, request):
        # Filter unpublished projects
        unpublished_projects = Project.objects.filter(is_published=False)

        # Create the HTTP response with CSV content type
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="youtube_titles.csv"'

        writer = csv.writer(response)
        writer.writerow(['Title Part 1', 'Title Part 2', 'Title Part 3', 
                         'Title Part 4', 'Title Part 5', 'Title Part 6', 
                         'Title Part 7', 'Title Part 8', 'Title Part 9'])  # Header row

        # Initialize a list to hold title parts
        title_parts_list = []

        for project in unpublished_projects:
            if hasattr(project, 'youtube_details') and project.youtube_details:
                title = project.youtube_details.thumbnail_title
                # Split the title into three parts
                title_parts = title.split(' ', 2)  # Split into at most 3 parts
                # Fill with empty strings if there are less than 3 parts
                title_parts += [''] * (3 - len(title_parts))
                title_parts_list.extend(title_parts)  # Add to the list

        # Write the title parts to the CSV in sets of 9 columns
        for i in range(0, len(title_parts_list), 9):
            writer.writerow(title_parts_list[i:i + 9])  # Write 9 columns at a time

        return response




openai.api_key = settings.OPENAI_API_KEY

class GenerateThumbnailView(View):
    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id)
        youtube_details = project.youtube_details

        if youtube_details and youtube_details.thumbnail_prompt:
            # Use project's video format for thumbnail
            aspect_ratio = "9:16" if project.video_format == 'reel' else "16:9"
            width, height = (1080, 1920) if project.video_format == 'reel' else (1280, 720)

            output = replicate_run(
                "black-forest-labs/flux-schnell",
                input={
                    "seed": 5,
                    "prompt": youtube_details.thumbnail_prompt,
                    "num_outputs": 1,
                    "aspect_ratio": aspect_ratio,
                    "output_format": "png",
                    "output_quality": 80,
                },
            )
            image_url = output[0]
            image_data = requests.get(image_url).content

            # Create a PIL Image from the image data
            image = Image.open(io.BytesIO(image_data))

            # Resize the image to appropriate dimensions
            image = image.resize((width, height), Image.LANCZOS)

            # Add thumbnail title text to the image
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("path/to/your/font.ttf", 60)
            except IOError:
                font = ImageFont.load_default()
            text = youtube_details.thumbnail_title
            text_width, text_height = draw.textsize(text, font=font)
            
            # Adjust text position based on format
            if project.video_format == 'reel':
                position = ((width - text_width) // 2, height - text_height - 100)
            else:
                position = ((width - text_width) // 2, height - text_height - 20)
                
            draw.text(position, text, font=font, fill=(255, 255, 255))

            # Save the image to a buffer
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            # Save the image to the YouTubeDetails model
            youtube_details.thumbnail.save(f"thumbnail_{project.id}.png", ContentFile(buffer.getvalue()))
            youtube_details.save()

        return redirect('home')
# Utility functions
def transcribe_audio_with_whisper(audio_path, transcription_path):
    if os.path.exists(transcription_path):
        with open(transcription_path, "r") as f:
            words = json.load(f)
        print(f"Loaded existing transcription from {transcription_path}")
    else:
        client = openai.OpenAI(api_key=openai.api_key)
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        words = transcription.words
        with open(transcription_path, "w") as f:
            json.dump(words, f, indent=4)
        print(f"Transcription saved to {transcription_path}")
    return words


def save_image(image_data, filename):
    with open(filename, "wb") as file:
        file.write(image_data)
    print(f"Image saved as {filename}")


# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(client, messages, tools=None, model="gpt-4o-mini"):
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        if tools:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
        else:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
        return response
    except Exception as e:
        print(f"Unable to generate ChatCompletion response: {e}")
        return e


def extract_youtube_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        return url.split("v=")[-1].split("&")[0]
    return None


def get_smartproxy_session():
    username = os.getenv('SMARTPROXY_USERNAME')
    password = os.getenv('SMARTPROXY_PASSWORD')
    endpoint = os.getenv('SMARTPROXY_ENDPOINT', 'gate.smartproxy.com')
    port = os.getenv('SMARTPROXY_PORT', '7000')

    proxy_url = f"http://{endpoint}:{port}"
    auth = HTTPProxyAuth(username, password)
    
    session = requests.Session()
    session.proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    session.auth = auth
    
    return session

 
def get_or_create_transcription(youtube_id):
    try:
        project = Project.objects.get(youtube_id=youtube_id)
        if project.transcription:
            return project.transcription
    except Project.DoesNotExist:
        project = None

    try:
        # Construct the YouTube URL from the ID
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        # Use your custom endpoint
        transcript_url = f"https://cjsubtitle.ananth-c-jayan.workers.dev/api/transcript?url={youtube_url}&output=json"
        
        response = requests.get(transcript_url)
        response.raise_for_status()  # Raise exception for bad status codes
        
        transcript_data = response.json()
        # Filter out [Music] entries and join the text
        transcription = " ".join(
            entry["text"] for entry in transcript_data 
            if entry["text"] and entry["text"] != "[Music]"
        )

        if project:
            project.transcription = transcription
            project.save()

        return transcription
    except Exception as e:
        print(f"Failed to get transcription: {str(e)}")
        raise


def generate_narration(transcript):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that creates engaging and original YouTube video scripts.",
        },
        {
            "role": "user",
            "content": f"""Create a modified and complete YouTube video script from this content: '{transcript}'.
           Divide the script into scenes, with each scene starting with 'Scene X:' where X is the scene number.""",
        },
    ]

    response = chat_completion_request(messages)

    if isinstance(response, Exception):
        raise Exception(f"Failed to generate narration: {response}")

    return response.choices[0].message.content


def create_scenes(project, narration):
    scenes = narration.split("Scene")[1:]
    for i, scene_content in enumerate(scenes, start=1):
        scene_lines = scene_content.strip().split("\n")
        scene_text = "\n".join(scene_lines[1:])
        Scene.objects.create(
            project=project,
            order=i,
            narration=scene_text,
            image_prompt=f"Generate an image for scene {i} of the video: {project.title}",
        )


def create_youtube_details(project):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that creates engaging YouTube video details.",
            },
            {
                "role": "user",
                "content": f"Create a title, description, thumbnail title, and thumbnail prompt for a YouTube video about: {project.title}",
            },
        ],
    )
    details = response.choices[0].message["content"].split("\n")
    YouTubeDetails.objects.create(
        project=project,
        title=details[0].replace("Title: ", ""),
        description=details[1].replace("Description: ", ""),
        thumbnail_title=details[2].replace("Thumbnail Title: ", ""),
        thumbnail_prompt=details[3].replace("Thumbnail Prompt: ", ""),
    )


def generate_voice(content, index, directory):
    filename = f"scene_{index}_audio.mp3"
    file_path = os.path.join(directory, filename)
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=content)
    response.stream_to_file(file_path)
    return file_path



def create_word_clip(word, start_time, duration, clip_width, clip_height, font_size=None):
    print(f"Creating word clip for: {word}")
    print(f"Dimensions: {clip_width}x{clip_height}")
    
    # Adjust font size based on video format
    if clip_height > clip_width:  # It's a reel
        font_size = font_size or 80  # Slightly smaller font for reels
        max_width = int(clip_width * 0.9)  # 90% of video width for reels
    else:
        font_size = font_size or 50
        max_width = int(clip_width * 0.8)  # 80% of video width for landscape

    try:
        # Split long text into multiple lines if needed
        words = word.split()
        lines = []
        current_line = []
        
        temp_clip = TextClip(
            "test",
            fontsize=font_size,
            color="white",
            font="Poppins-Bold",
            stroke_color='black',
            stroke_width=2,
        )
        
        current_width = 0
        for w in words:
            test_clip = TextClip(
                w,
                fontsize=font_size,
                color="white",
                font="Poppins-Bold",
                stroke_color='black',
                stroke_width=2,
            )
            word_width = test_clip.w
            test_clip.close()
            
            if current_width + word_width <= max_width:
                current_line.append(w)
                current_width += word_width + font_size//2  # Add space between words
            else:
                lines.append(" ".join(current_line))
                current_line = [w]
                current_width = word_width
        
        if current_line:
            lines.append(" ".join(current_line))
        
        # Create text clip for each line
        text_clips = []
        total_height = 0
        for line in lines:
            line_clip = TextClip(
                line,
                fontsize=font_size,
                color="white",
                font="Poppins-Bold",
                stroke_color='black',
                stroke_width=2,
                method='label',
            )
            text_clips.append(line_clip)
            total_height += line_clip.h + 10  # Add spacing between lines
        
        # Create background
        bg_height = total_height + 40  # Add padding
        bg_clip = ColorClip(
            size=(clip_width, bg_height),
            color=(0, 0, 0)
        ).set_opacity(0.7)
        
        # Position text clips vertically
        y_position = 20
        positioned_clips = [bg_clip]
        for text_clip in text_clips:
            positioned_clip = text_clip.set_position(
                ('center', y_position)
            )
            positioned_clips.append(positioned_clip)
            y_position += text_clip.h + 10
        
        # Composite all clips
        final_txt_clip = (CompositeVideoClip(
            positioned_clips,
            size=(clip_width, bg_height)
        )
        .set_start(start_time)
        .set_duration(duration + 0.1)
        .crossfadein(0.1))
        
        return final_txt_clip
    except Exception as e:
        print(f"Error creating word clip: {e}")
        return None


def resize_with_aspect_ratio(image_path, max_size):
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(image_path)


def create_scenes_and_youtube_details(project, transcription):
    client = openai.OpenAI(api_key=openai.api_key)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_scenes_and_narration",
                "description": """this function generates an engaging narration for a part of the 
                story which is given by parameter content and then generates Disney character style images for that corresponding narration. 
                The prompt parameter should describe each person and scene in a very detailed way. Never use any names,
                  always describe the gender, facial features, and dress consistently throughout the story. 
                Each prompt is independent, so we need to describe the person and situation 
                each time while maintaining consistency.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "This describes the narration content of the scene , it should be in simple words and should be it should be less than 200 words",
                        },
                        "prompt": {"type": "string"},
                        "mood": {
                            "type": "string",
                            "description": "This describes the mood of the scene from options: [adventure, dramatic, happy, romantic, suspense]",
                        },
                    },
                    "required": ["content", "prompt", "mood"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_youtube_details",
                "description": """This function generates an engaging YouTube title, description, and thumbnail prompt.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "A catchy, SEO-optimized title for the story on YouTube.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "A prompt for an image model to generate a catchy thumbnail for the YouTube video.",
                        },
                        "thumbnail_title": {
                            "type": "string",
                            "description": "A short, exactly 4-word title for the thumbnail text, designed to attract clicks.",
                        },
                        "description": {
                            "type": "string",
                            "description": "An SEO-optimized description targeting keywords relevant to kids' stories.",
                        },
                    },
                    "required": [
                        "title",
                        "thumbnail_title",
                        "description",
                        "prompt",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that creates engaging and original YouTube video scripts.",
        },
        {
            "role": "user",
            "content": f"""Create a modified and complete YouTube video script from this content: '{transcription}'. 
            We should remove the youtube channel name , or speakers or authors name of the vedio if it includes the same , 
            we should also remove the credits, 
            but if the name is not of the youtube narrator 
            or any credits and if its related to story we should include the name 
            Make sure to use a strong hook at the beginning to engage viewers. 
            The script should be original and free of plagiarism.
            The script should not be like it is copied from this vedio,
            Generate narration for each scene ,then generate images, and other necessary assets for the scenes. 
            atleast some 8 scenes should be there in the video and
            Each image prompt should be self-explanatory and consistent in 
            describing the characters and scenes without assuming prior knowledge of the content. 
            Please make sure that  characters are described as same citizen of the country 
            in details mentioned and are ANIMATED also avoid texts in images.
            Also, generate a compelling YouTube title, description, and thumbnail title for the video.""",
        },
    ]

    response = chat_completion_request(client, messages=messages, tools=tools)
    print(response)

    # Process the response
    scenes = []
    youtube_details = None

    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "generate_scenes_and_narration":
            scenes.append(
                {
                    "narration": arguments.get("content"),
                    "image_prompt": arguments.get("prompt"),
                    "mood": arguments.get("mood"),
                }
            )
        elif function_name == "generate_youtube_details":
            youtube_details = arguments

    # Create scenes
    for i, scene in enumerate(scenes, start=1):
        Scene.objects.create(
            project=project,
            order=i,
            narration=scene["narration"],
            image_prompt=scene["image_prompt"],
            mood=scene["mood"],
        )

    # Create YouTube details
    if youtube_details:
        YouTubeDetails.objects.create(
            project=project,
            title=youtube_details["title"],
            description=youtube_details["description"],
            thumbnail_title=youtube_details["thumbnail_title"],
            thumbnail_prompt=youtube_details["prompt"],
        )


# Views
class HomeView(View):
    def get(self, request):
        projects = Project.objects.all()

        # Apply filters
        search_query = request.GET.get('search')
        status = request.GET.get('status')
        tag = request.GET.get('tag')
        has_youtube_details = request.GET.get('has_youtube_details')
        has_audio = request.GET.get('has_audio')
        has_image = request.GET.get('has_image')
        has_final_video = request.GET.get('has_final_video')
        is_published = request.GET.get('is_published')  # New filter

        if not (search_query or status or tag or has_youtube_details or has_audio or has_image or has_final_video or is_published):
            projects = projects.filter(is_published=False)
        if search_query:
            projects = projects.filter(Q(title__icontains=search_query) | Q(youtube_url__icontains=search_query))

        if status:
            projects = projects.filter(status=status)

        if tag:
            projects = projects.filter(tag=tag)

        if has_youtube_details:
            if has_youtube_details == 'yes':
                projects = projects.filter(youtube_details__isnull=False)
            elif has_youtube_details == 'no':
                projects = projects.filter(youtube_details__isnull=True)

        if has_audio:
            if has_audio == 'yes':
                projects = projects.filter(scenes__audio__isnull=False).distinct()
            elif has_audio == 'no':
                projects = projects.exclude(scenes__audio__isnull=False)

        if has_image:
            if has_image == 'yes':
                projects = projects.filter(scenes__image__isnull=False).distinct()
            elif has_image == 'no':
                projects = projects.exclude(scenes__image__isnull=False)

        if has_final_video:
            if has_final_video == 'yes':
                projects = projects.filter(final_video__isnull=False)
            elif has_final_video == 'no':
                projects = projects.filter(final_video__isnull=True)

        if is_published:  # Filter by published status
            if is_published == 'yes':
                projects = projects.filter(is_published=True)
            elif is_published == 'no':
                projects = projects.filter(is_published=False)

        return render(request, "narration_app/home.html", {"projects": projects})


class CreateNarrationView(View):
    def get(self, request):
        form = ProjectForm()
        return render(request, "narration_app/create_narration.html", {"form": form})

    def post(self, request):
        form = ProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            youtube_id = extract_youtube_id(project.youtube_url)
            project.youtube_id = youtube_id
            project.save()

            # Start the Celery task
            create_narration_task.delay(project.id)

            return redirect("edit_narration", project_id=project.id)
        return render(request, "narration_app/create_narration.html", {"form": form})


class EditNarrationView(View):
    def get(self, request, project_id):
        # Redirect directly to EditImagesView
        return redirect("edit_images", project_id=project_id)

    def post(self, request, project_id):
        # This method can be removed if you're not using it anymore
        pass


class EditImagesView(View):
    def get(self, request, project_id):
        project = get_object_or_404(Project, id=project_id)
        scenes = project.scenes.all().order_by("order")
        return render(
            request,
            "narration_app/edit_images.html",
            {"project": project, "scenes": scenes},
        )

    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id)
        scenes = project.scenes.all()

        if 'create_all_images' in request.POST:
            for scene in scenes:
                generate_scene_image_task.delay(scene.id, project.id)
            return redirect("edit_images", project_id=project.id)

        if 'create_all_audios' in request.POST:
            for scene in scenes:
                generate_scene_audio_task.delay(scene.id, project.id)
            return redirect("edit_images", project_id=project.id)

        if 'generate_final_video' in request.POST:
            return redirect('generate_video', project_id=project.id)

        for scene in scenes:
            scene.narration = request.POST.get(f"narration_{scene.id}")
            scene.image_prompt = request.POST.get(f"image_prompt_{scene.id}")
            
            if f"regenerate_{scene.id}" in request.POST:
                generate_scene_image_task.delay(scene.id, project.id)
            
            if f"generate_audio_{scene.id}" in request.POST:
                generate_scene_audio_task.delay(scene.id, project.id)
            
            if f"custom_image_{scene.id}" in request.FILES:
                custom_image = request.FILES[f"custom_image_{scene.id}"]
                custom_image_name = f"scene_{scene.id}_{custom_image.name}"
                scene.image.save(custom_image_name, custom_image)
            
            scene.save()

        return redirect("edit_images", project_id=project.id)


class GenerateVideoView(View):
    def get(self, request, project_id):
        project = get_object_or_404(Project, id=project_id)
        return render(
            request, "narration_app/generate_video.html", {"project": project}
        )

    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id)
        
        # Start the video generation task
        generate_video_task.delay(project.id)
        
        return redirect("home")


class DeleteProjectView(View):
    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id)
        project.delete()
        return redirect('home')


class RegenerateProjectView(View):
    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id)
        
        # Clear existing scenes and YouTube details
        project.scenes.all().delete()
        if hasattr(project, 'youtube_details'):
            project.youtube_details.delete()
        
        # Re-run the narration generation process
        transcription = get_or_create_transcription(project.youtube_id)
        
        create_scenes_and_youtube_details(project, transcription)
        
        return redirect('edit', project_id=project.id)


class GenerateVideosForAllProjectsView(View):
    def get(self, request):
        from .tasks import generate_videos_for_all_projects
        generate_videos_for_all_projects.delay()
        return redirect("home")


class BulkCreateProjectsView(View):
    def get(self, request):
        return render(request, 'narration_app/bulk_create_projects.html')

    def post(self, request):
        from .tasks import process_youtube_url
        csv_file = request.FILES['csv_file']
        content = csv_file.read().decode('utf-8')
        csv_data = csv.reader(StringIO(content))
        
        # Skip header row if it exists
        next(csv_data, None)
        
        for row in csv_data:
            if len(row) >= 2:
                youtube_url, title = row[0], row[1]
                process_youtube_url.delay(youtube_url, title)
        
        return redirect('home')


class CreateCustomNarrationView(FormView):
    template_name = 'narration_app/create_custom_narration.html'
    form_class = CustomScriptForm
    success_url = '/edit_narration/{project_id}/'

    def form_valid(self, form):
        custom_script = form.cleaned_data['custom_script']
        project = Project.objects.create(title="Custom Script Project", youtube_url="")

        client = openai.OpenAI(api_key=openai.api_key)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_scenes_and_narration",
                    "description": """This function generates an engaging narration for a part of the 
                    story which is given by parameter content and then generates Disney character style images for that corresponding narration. 
                    The prompt parameter should describe each person and scene in a very detailed way. Never use any names,
                    always describe the gender, facial features, and dress consistently throughout the story. 
                    Each prompt is independent, so we need to describe the person and situation 
                    each time while maintaining consistency.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "This describes the narration content of the scene, it should be in simple words and should be less than 200 words",
                            },
                            "prompt": {"type": "string"},
                            "mood": {
                                "type": "string",
                                "description": "This describes the mood of the scene from options: [adventure, dramatic, happy, romantic, suspense]",
                            },
                        },
                        "required": ["content", "prompt", "mood"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_youtube_details",
                    "description": """This function generates an engaging YouTube title, description, and thumbnail prompt.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "A catchy, SEO-optimized title for the story on YouTube.",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "A prompt for an image model to generate a catchy thumbnail for the YouTube video.",
                            },
                            "thumbnail_title": {
                                "type": "string",
                                "description": "A short, exactly 4-word title for the thumbnail text, designed to attract clicks.",
                            },
                            "description": {
                                "type": "string",
                                "description": "An SEO-optimized description targeting keywords relevant to kids' stories.",
                            },
                        },
                        "required": [
                            "title",
                            "thumbnail_title",
                            "description",
                            "prompt",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        ]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates engaging and original YouTube video scripts.",
            },
            {
                "role": "user",
                "content": f"""Create a modified and complete YouTube video script from this idea: '{custom_script}'. 
                Make sure to use a strong hook at the beginning to engage viewers. 
                The script should be original and free of plagiarism.
                Generate narration for each scene, then generate images, and other necessary assets for the scenes. 
                Each image prompt should be self-explanatory and consistent in 
                describing the characters and scenes without assuming prior knowledge of the content. 
                Please make sure that characters are described as same citizen of the country 
                in details mentioned and are ANIMATED also avoid texts in images.
                Also, generate a compelling YouTube title, description, and thumbnail title for the video.""",
            },
        ]

        response = chat_completion_request(client, messages=messages, tools=tools)

        scenes = []
        youtube_details = None

        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "generate_scenes_and_narration":
                scenes.append(
                    {
                        "narration": arguments.get("content"),
                        "image_prompt": arguments.get("prompt"),
                        "mood": arguments.get("mood"),
                    }
                )
            elif function_name == "generate_youtube_details":
                youtube_details = arguments

        for i, scene in enumerate(scenes, start=1):
            Scene.objects.create(
                project=project,
                order=i,
                narration=scene["narration"],
                image_prompt=scene["image_prompt"],
                mood=scene["mood"],
            )

        if youtube_details:
            YouTubeDetails.objects.create(
                project=project,
                title=youtube_details["title"],
                description=youtube_details["description"],
                thumbnail_title=youtube_details["thumbnail_title"],
                thumbnail_prompt=youtube_details["prompt"],
            )

        self.success_url = self.success_url.format(project_id=project.id)
        return super().form_valid(form)        # Add this new view

# Add this helper function at the top with other utility functions
def get_image_dimensions(video_format):
    if video_format == 'reel':
        return (1080, 1920)  # Standard Reel dimensions
    return (1920, 1080)  # Standard landscape dimensions

class UpdatePublishedStatusView(View):
    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id)
        data = json.loads(request.body)
        project.is_published = data.get('is_published', False)
        project.save()
        return JsonResponse({'status': 'success'})
