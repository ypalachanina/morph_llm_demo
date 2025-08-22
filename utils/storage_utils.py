import os
import streamlit as st
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta


class StorageClient:
    def __init__(self, secrets):
        self.connection_string = secrets["CONNECTION_STRING"]
        self.video_client = self.get_blob_container_client()

    def get_blob_container_client(self, container_name="videos"):
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        return container_client

    def load_model_weights(self, weights, save_to_root=True):
        model_client = self.get_blob_container_client("models")
        for blob_name in weights.values():
            if not os.path.exists(blob_name):
                blob_client = model_client.get_blob_client(blob_name)
                blob_data = blob_client.download_blob().readall()
                if save_to_root:
                    with open(blob_name, "wb") as download_file:
                        download_file.write(blob_data)

    def list_azure_videos(self):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        videos = {}
        blob_list = self.video_client.list_blobs(name_starts_with="streamlit_videos/")
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

    def get_video_url(self, blob_name):
        try:
            conn_dict = {}
            for part in self.connection_string.split(';'):
                if '=' in part:
                    key, value = part.split('=', 1)
                    conn_dict[key] = value

            account_name = conn_dict.get('AccountName')
            account_key = conn_dict.get('AccountKey')
            container_name = self.video_client.container_name

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
