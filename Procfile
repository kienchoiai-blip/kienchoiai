web: gunicorn app:app --timeout 1200 --graceful-timeout 1200 --workers 1 --threads 1 --worker-class sync --preload-app false --max-requests 10 --max-requests-jitter 5 --keep-alive 5

