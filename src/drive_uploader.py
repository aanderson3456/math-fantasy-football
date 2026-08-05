import os
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

def upload_file_to_drive(file_path, mime_type="text/csv", folder_id=None):
    """Uploads a file to Google Drive using local Application Default Credentials (ADC).
    
    Args:
        file_path (str): The local path to the file to upload.
        mime_type (str): The MIME type of the file. Defaults to "text/csv".
        folder_id (str, optional): The ID of the parent folder in Google Drive.
        
    Returns:
        dict: A dictionary containing 'success' (bool), 'file_id' (str or None), and 'message' (str).
    """
    if not os.path.exists(file_path):
        return {
            'success': False,
            'file_id': None,
            'message': f"Error: Local file '{file_path}' does not exist."
        }
        
    filename = os.path.basename(file_path)
    
    try:
        # Load default credentials. Will search for ADC in standard locations.
        creds, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        
        # Build the Drive API service (v3)
        service = build('drive', 'v3', credentials=creds)

        # Define file metadata
        file_metadata = {'name': filename}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        # Specify the media to upload
        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

        # Execute the upload
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name'
        ).execute()

        file_id = file.get('id')
        return {
            'success': True,
            'file_id': file_id,
            'message': f"Successfully uploaded '{filename}' to Google Drive (ID: {file_id})"
        }

    except google.auth.exceptions.DefaultCredentialsError:
        return {
            'success': False,
            'file_id': None,
            'message': "Google Application Default Credentials (ADC) not found. Run 'gcloud auth application-default login' in your terminal."
        }
    except HttpError as error:
        return {
            'success': False,
            'file_id': None,
            'message': f"Google Drive API Error: {error.reason}"
        }
    except Exception as e:
        return {
            'success': False,
            'file_id': None,
            'message': f"Unexpected error occurred: {str(e)}"
        }
