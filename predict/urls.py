from django.urls import path
from .views import *


urlpatterns = [
    path('', PredictionView.as_view()),
]