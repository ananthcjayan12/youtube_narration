from django.contrib import admin
from .models import Project, Scene, YouTubeDetails


class SceneInline(admin.TabularInline):
    model = Scene
    extra = 1
    fields = ("order", "narration", "image_prompt", "mood", "image")
    readonly_fields = ("image",)


class YouTubeDetailsInline(admin.StackedInline):
    model = YouTubeDetails
    fields = (
        "title",
        "description",
        "thumbnail_title",
        "thumbnail_prompt",
        "thumbnail_image",
    )
    readonly_fields = ("thumbnail_image",)


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "youtube_url",
        "youtube_id",
        "scene_count",
        "has_final_video",
        "has_youtube_details",
    )
    list_filter = ("scenes__mood",)  # Removed 'youtube_details__isnull'
    search_fields = ("title", "youtube_id", "transcription")
    readonly_fields = ("youtube_id", "transcription")
    inlines = [SceneInline, YouTubeDetailsInline]
    actions = ['remove_final_video']

    def scene_count(self, obj):
        return obj.scenes.count()

    scene_count.short_description = "Number of Scenes"

    def has_final_video(self, obj):
        return bool(obj.final_video)

    has_final_video.boolean = True
    has_final_video.short_description = "Final Video"

    def has_youtube_details(self, obj):
        return hasattr(obj, "youtube_details")

    has_youtube_details.boolean = True
    has_youtube_details.short_description = "Has YouTube Details"

    def remove_final_video(self, request, queryset):
        updated = queryset.update(final_video='')
        self.message_user(request, f'{updated} projects had their final video removed.')

    remove_final_video.short_description = "Remove final video"


@admin.register(Scene)
class SceneAdmin(admin.ModelAdmin):
    list_display = ("project", "order", "mood", "narration_preview", "has_image")
    list_filter = ("project", "mood")
    search_fields = ("project__title", "narration", "image_prompt")
    readonly_fields = ("image",)
    ordering = ("project", "order")

    def narration_preview(self, obj):
        return obj.narration[:50] + "..." if len(obj.narration) > 50 else obj.narration

    narration_preview.short_description = "Narration Preview"

    def has_image(self, obj):
        return bool(obj.image)

    has_image.boolean = True
    has_image.short_description = "Image"


@admin.register(YouTubeDetails)
class YouTubeDetailsAdmin(admin.ModelAdmin):
    list_display = ("project", "title", "description_preview", "has_thumbnail")
    search_fields = (
        "project__title",
        "title",
        "description",
        "thumbnail_title",
        "thumbnail_prompt",
    )
    readonly_fields = ("thumbnail_image",)

    def description_preview(self, obj):
        return (
            obj.description[:50] + "..."
            if len(obj.description) > 50
            else obj.description
        )

    description_preview.short_description = "Description Preview"

    def has_thumbnail(self, obj):
        return bool(obj.thumbnail_image)

    has_thumbnail.boolean = True
    has_thumbnail.short_description = "Thumbnail"
