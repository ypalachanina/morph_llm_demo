import os
import streamlit as st
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

# CONNECTION_STRING = os.environ.get("CONNECTION_STRING")
# CONNECTION_STRING = os.getenv("CONNECTION_STRING")
CONNECTION_STRING = st.secrets("CONNECTION_STRING")


def get_blob_container_client():
    container_name = "videos"
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(container_name)
    return container_client


@st.cache_data(ttl=300)
def list_azure_videos():
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    videos = {}
    container_client = get_blob_container_client()
    blob_list = container_client.list_blobs(name_starts_with="streamlit_videos/")
    for blob in blob_list:
        if any(blob.name.lower().endswith(ext) for ext in video_extensions):
            video = {
                'name': blob.name,
                'display_name': os.path.basename(blob.name),
                'size': blob.size,
                'size_mb': f"{blob.size / (1024 * 1024):.1f} MB",
                'last_modified': blob.last_modified
            }
            video['desc'] = f"{video['display_name']} ({video['size_mb']})"
            videos[video['display_name']] = video
    return videos


def get_video_url(blob_name):
    try:
        container_client = get_blob_container_client()
        conn_dict = {}
        for part in CONNECTION_STRING.split(';'):
            if '=' in part:
                key, value = part.split('=', 1)
                conn_dict[key] = value

        account_name = conn_dict.get('AccountName')
        account_key = conn_dict.get('AccountKey')
        container_name = container_client.container_name

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )

        blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        return blob_url

    except Exception as e:
        st.error(f"Error generating video URL: {e}")
        return None
