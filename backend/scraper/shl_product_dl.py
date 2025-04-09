import requests
import os
import time
import csv
import re
from urllib.parse import urlparse


def sanitize_filename(url):
    """
    Creates a safe filename from a URL by extracting the product name from the path
    """
    parsed_url = urlparse(url)
    product_name = parsed_url.path.split("/")[-2]

    sanitized_name = re.sub(r"[^\w\-]", "_", product_name)
    return sanitized_name


def download_shl_product_pages(csv_path):
    """
    Downloads all SHL product pages listed in the CSV file
    """
    save_dir = "shl_catalog_product_pages"
    os.makedirs(save_dir, exist_ok=True)

    # Headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Read the CSV file
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        # Count total number of products for progress reporting
        product_count = sum(1 for _ in reader)
        csvfile.seek(0)  # Reset file pointer
        next(reader)  # Skip header again

        print(f"Found {product_count} products to download.")

        # Process each product
        for i, row in enumerate(reader, 1):
            product_name = row["Name"]
            product_url = row["Link"]

            # Create a filename based on the URL
            filename = sanitize_filename(product_url) + ".html"
            file_path = os.path.join(save_dir, filename)

            # Skip if already downloaded
            if os.path.exists(file_path):
                print(
                    f"[{i}/{product_count}] Skipping (already exists): {product_name}"
                )
                continue

            print(
                f"[{i}/{product_count}] Downloading: {product_name} from {product_url}"
            )

            try:
                # Make the request
                response = requests.get(product_url, headers=headers)
                response.raise_for_status()

                # Check if the response is HTML
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    print(
                        f"Warning: Response is not HTML. Content-Type: {content_type}"
                    )

                # Save the file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(response.text)

                print(f"  → Saved to {file_path}")

                # Add a small delay to be nice to the server
                time.sleep(1.5)

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {product_name}: {e}")

            # Every 50 requests, take a longer break
            if i % 50 == 0:
                print(f"Taking a break after {i} downloads...")
                time.sleep(10)

    print(f"\nDownload complete. Product pages saved to '{save_dir}' directory.")
    print(f"Total products attempted: {product_count}")


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "shl_solutions.csv")
    download_shl_product_pages(csv_path)
