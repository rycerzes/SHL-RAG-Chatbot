import requests
import os
import time


def download_shl_catalog():
    save_dir = "shl_catalog_pages"
    os.makedirs(save_dir, exist_ok=True)

    base_url = "https://www.shl.com/solutions/products/product-catalog/"

    # Define the page parameters
    # ?start=0&type=1&type=1
    # ?start=372&type=1&type=1
    # 12 pages in between

    total_pages = 32
    last_page_start = 372

    step_size = last_page_start // (total_pages - 1)

    start_values = [i * step_size for i in range(total_pages)]
    start_values[-1] = 372

    # headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    for i, start in enumerate(start_values):
        page_number = i + 1
        url = f"{base_url}?start={start}&type=1&type=1"

        print(f"Downloading page {page_number}/{total_pages}: {url}")

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                print(f"Warning: Response is not HTML. Content-Type: {content_type}")

            file_path = os.path.join(save_dir, f"shl_catalog_page_{page_number}.html")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            print(f"Saved page {page_number} to {file_path}")

            time.sleep(2)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading page {page_number}: {e}")

    print(f"Download complete. Pages saved to '{save_dir}' directory.")


if __name__ == "__main__":
    download_shl_catalog()
