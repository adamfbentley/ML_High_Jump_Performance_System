import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

VIDEO_INGESTION_SERVICE_URL = os.getenv("VIDEO_INGESTION_SERVICE_URL", "http://video_ingestion_service:8002/internal")
SESSION_BVH_DATA_SERVICE_URL = os.getenv("SESSION_BVH_DATA_SERVICE_URL", "http://session_bvh_data_service:8003/internal")
POSE_ESTIMATION_SERVICE_URL = os.getenv("POSE_ESTIMATION_SERVICE_URL", "http://pose_estimation_service:8006/internal")
PINN_GNN_INFERENCE_SERVICE_URL = os.getenv("PINN_GNN_INFERENCE_SERVICE_URL", "http://pinn_gnn_inference_service:8008/internal") # NEW
ANOMALY_DETECTION_SERVICE_URL = os.getenv("ANOMALY_DETECTION_SERVICE_URL", "http://anomaly_detection_service:8009/internal") # NEW
