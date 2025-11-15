"""
Script to upload local parquet files to S3
Usage: python upload_data.py <parquet_file> [bucket_name]
"""

import sys
import os
import boto3
from pathlib import Path

def upload_to_s3(local_file, bucket_name=None, region='us-east-2'):
    """
    Upload a parquet file to S3
    
    Args:
        local_file: Path to local parquet file
        bucket_name: S3 bucket name (optional, will use default if not provided)
        region: AWS region
    """
    s3_client = boto3.client('s3', region_name=region)
    
    #get account ID
    if bucket_name is None:
        sts = boto3.client('sts', region_name=region)
        account_id = sts.get_caller_identity()['Account']
        bucket_name = f'stock-pipeline-data-{account_id}'
    
    #file exists
    if not os.path.exists(local_file):
        raise FileNotFoundError(f"File not found: {local_file}")
    
    filename = Path(local_file).name
    
    #raw data key
    s3_key = f'raw/{filename}'
    
    print(f"Uploading {local_file} to s3://{bucket_name}/{s3_key}")
    
    try:
        s3_client.upload_file(
            local_file,
            bucket_name,
            s3_key,
            ExtraArgs={
                'ServerSideEncryption': 'AES256',
                'Metadata': {
                    'uploaded-by': 'upload_script',
                    'original-filename': filename
                }
            }
        )
        
        print(f"Successfully uploaded to S3!")
        print(f"\nBucket: {bucket_name}")
        print(f"\nKey: {s3_key}")
        print(f"\nNext:")
        print(f"aws glue start-crawler --name stock-pipeline-raw-crawler")
        print(f"glue start-job-run --job-name stock-pipeline-clean-job")
        return True
        
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python upload_data.py <local_parquet_file> [bucket_name]")
        print("\nExample:")
        print("  python upload_data.py stock_data.parquet")
        print("  python upload_data.py stock_data.parquet my-custom-bucket")
        sys.exit(1)
    
    local_file = sys.argv[1]
    bucket_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = upload_to_s3(local_file, bucket_name)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()