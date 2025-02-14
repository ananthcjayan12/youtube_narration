import os
import json
import openai
import requests
from django.conf import settings
from PIL import Image
from moviepy.editor import TextClip, ColorClip, CompositeVideoClip
from requests.auth import HTTPProxyAuth

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
        from .models import Project
        project = Project.objects.get(youtube_id=youtube_id)
        if project.transcription:
            return project.transcription
    except Project.DoesNotExist:
        project = None

    try:
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        transcript_url = f"https://cjsubtitle.ananth-c-jayan.workers.dev/api/transcript?url={youtube_url}&output=json"
        
        response = requests.get(transcript_url)
        response.raise_for_status()
        
        transcript_data = response.json()
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
    
    if clip_height > clip_width:
        font_size = font_size or 80
        max_width = int(clip_width * 0.9)
    else:
        font_size = font_size or 50
        max_width = int(clip_width * 0.8)

    try:
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
                current_width += word_width + font_size//2
            else:
                lines.append(" ".join(current_line))
                current_line = [w]
                current_width = word_width
        
        if current_line:
            lines.append(" ".join(current_line))
        
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
            total_height += line_clip.h + 10
        
        bg_height = total_height + 40
        bg_clip = ColorClip(
            size=(clip_width, bg_height),
            color=(0, 0, 0)
        ).set_opacity(0.7)
        
        y_position = 20
        positioned_clips = [bg_clip]
        for text_clip in text_clips:
            positioned_clip = text_clip.set_position(
                ('center', y_position)
            )
            positioned_clips.append(positioned_clip)
            y_position += text_clip.h + 10
        
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
    from .models import Scene, YouTubeDetails
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

def get_image_dimensions(video_format):
    if video_format == 'reel':
        return (1080, 1920)  # Standard Reel dimensions
    return (1920, 1080)  # Standard landscape dimensions 