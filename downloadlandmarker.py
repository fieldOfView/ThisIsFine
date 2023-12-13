import requests
import tqdm
import argparse

def download(url: str, filename: str):
    with open(filename, "wb") as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                "desc": url,
                "total": total,
                "miniters": 1,
                "unit": "B",
                "unit_scale": True,
                "unit_divisor": 1024,
            }
            with tqdm.tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["lite", "full", "heavy"], default="full", nargs="?")

    variant = parser.parse_args().variant

    download(
        f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task",
        "resources/pose_landmarker.task",
    )
