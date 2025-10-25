
import mimetypes
import os
import boto3
import streamlit as st

from common.logger import Logger 

LOGGER = Logger.get_logger("common.utils")

def filename_key(filename: str) -> str:
    """Return filename without extension as object key."""
    base = os.path.basename(filename or "").strip()
    stem, _ = os.path.splitext(base)
    return stem


def _s3_client_from_secrets():
    """
    Build boto3 client from Streamlit secrets.
    Expected st.secrets["s3"] keys:
      endpoint_url, region, aws_access_key_id, aws_secret_access_key, bucket, base_path (optional)
    """
    s3s = st.secrets["s3"]
    client = boto3.client(
        "s3",
        aws_access_key_id=s3s.get("aws_access_key_id"),
        aws_secret_access_key=s3s.get("aws_secret_access_key"),
        region_name=s3s.get("region"),
        endpoint_url=s3s.get("endpoint_url"),
    )
    return client


def upload_image_to_s3(image_bytes: bytes, object_key: str, filename: str) -> str:
    """
    Upload image bytes to S3 bucket at {base_path}/{object_key}.
    ContentType inferred from filename.
    Returns a public-style URL (endpoint/bucket/base_path/object_key).
    """
    try:
        s3s = st.secrets["s3"]
        client = _s3_client_from_secrets()
        bucket = s3s["bucket"]
        base_path = s3s.get("base_path", "").strip("/")
        content_type = mimetypes.guess_type(filename or "")[0] or "application/octet-stream"

        key = f"{base_path}/{object_key}" if base_path else object_key
        client.put_object(Bucket=bucket, Key=key, Body=image_bytes, ContentType=content_type)

        endpoint = s3s.get("public_base_url") or s3s.get("endpoint_url")
        # Build URL for convenience; adapt if you use CDN.
        url = f"{endpoint.rstrip('/')}/{bucket}/{key}"
        LOGGER.info("[S3] put_object ok bucket=%s key=%s", bucket, key)
        return url
    except Exception as e:
        LOGGER.exception("[S3][ERROR] %s", e)
        raise
