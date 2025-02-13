from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import (
    HomeView,
    CreateNarrationView,
    EditNarrationView,
    EditImagesView,
    GenerateVideoView,
    DeleteProjectView,
    RegenerateProjectView,
    CreateCustomNarrationView,
    GenerateVideosForAllProjectsView,
    BulkCreateProjectsView,
    GenerateThumbnailView,
    UpdatePublishedStatusView,
    TaskStatusView,
    GenerateCSVView
)

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("create/", CreateNarrationView.as_view(), name="create_narration"),
    path("edit/<int:project_id>/", EditNarrationView.as_view(), name="edit_narration"),
    path("edit_images/<int:project_id>/", EditImagesView.as_view(), name="edit_images"),
    path(
        "generate_video/<int:project_id>/",
        GenerateVideoView.as_view(),
        name="generate_video",
    ),
    path("delete/<int:project_id>/", DeleteProjectView.as_view(), name="delete_project"),
    path("regenerate/<int:project_id>/", RegenerateProjectView.as_view(), name="regenerate_project"),
    path('create_custom_narration/', CreateCustomNarrationView.as_view(), name='create_custom_narration'),
    path("generate_all_videos/", GenerateVideosForAllProjectsView.as_view(), name="generate_videos_for_all_projects"),
    path('bulk-create/', BulkCreateProjectsView.as_view(), name='bulk_create_projects'),
    path('generate_thumbnail/<int:project_id>/', GenerateThumbnailView.as_view(), name='generate_thumbnail'),
    path('update_published_status/<int:project_id>/', UpdatePublishedStatusView.as_view(), name='update_published_status'),
    path('task-status/', TaskStatusView.as_view(), name='task_status'),
    path('generate_csv/', GenerateCSVView.as_view(), name='generate_csv'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

