import os
import requests
from dotenv import load_dotenv
import base64


def get_geotiff_with_subet_api(resource_ern, bbox):

    url = f"https://api.eratos.com/api/workspace/v1/subset?resourceId={resource_ern}&bbox={bbox}&direction=-1&limit=1&skip=0"
    load_dotenv()
    eratos_key = os.getenv("ERATOS_KEY")
    eratos_secret = os.getenv("ERATOS_SECRET")
    credentials = f"{eratos_key}:{eratos_secret}"
    encoded_cred = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    headers = {
        "Accept": "image/tiff",
        "authorization": f"Basic {encoded_cred}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open("min_temp_subset.tif", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192): # read in 8kb per chunk
                f.write(chunk)
        print("GeoTIFF downloaded successfully.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == '__main__':
    resource_ern = "ern:e-pn.io:resource:fahma.blocks.daily.frost.metrics.2025.min.temp"

    bbox = "148.650169, -34.496229, 148.736687, -34.446698" # an example of bounding box
    get_geotiff_with_subet_api(
        resource_ern,
        bbox
    )