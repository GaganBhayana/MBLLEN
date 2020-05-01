import tqdm
from google.colab import auth
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload

def save_file_to_google_drive(local_filename, dest_filename, mimetype = 'application/octet-stream'):
  auth.authenticate_user()
  drive_service = build('drive', 'v3')

  file_metadata = {
    'name': dest_filename,
    'mimeType': mimetype
  }
  media = MediaFileUpload(local_filename, 
                          mimetype=mimetype,
                          resumable=True)
  created = drive_service.files().create(body=file_metadata,
                                         media_body=media,
                                         fields='id').execute()
  print('File ID: {}'.format(created.get('id')))
  return created.get('id')
