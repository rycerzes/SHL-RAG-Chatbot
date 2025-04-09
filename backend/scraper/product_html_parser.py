import os
import csv
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def sanitize_filename(url):
    """
    Creates a safe filename from a URL by extracting the product name from the path
    """
    parsed_url = urlparse(url)
    product_name = parsed_url.path.split("/")[-2]

    sanitized_name = re.sub(r"[^\w\-]", "_", product_name)
    return sanitized_name


def extract_product_details(html_content):
    """Extract description, languages, job levels, and completion time from the HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")

    description = None
    languages = None
    job_levels = None
    completion_time = None

    details_divs = soup.select("div.product-catalogue-training-calendar__row.typ")

    for div in details_divs:
        description_header = div.find("h4", string="Description")
        if description_header:
            description_p = description_header.find_next("p")
            if description_p:
                description = description_p.get_text(strip=True)

        languages_header = div.find("h4", string="Languages")
        if languages_header:
            languages_p = languages_header.find_next("p")
            if languages_p:
                languages = languages_p.get_text(strip=True)

        job_levels_header = div.find("h4", string="Job levels")
        if job_levels_header:
            job_levels_p = job_levels_header.find_next("p")
            if job_levels_p:
                job_levels = job_levels_p.get_text(strip=True)

        for p_tag in div.find_all("p"):
            p_text = p_tag.get_text(strip=True)
            if "Approximate Completion Time in minutes" in p_text:
                time_match = re.search(r"=\s*([^,]+)", p_text)
                if time_match:
                    completion_time = time_match.group(1).strip()
                else:
                    time_match = re.search(r"minutes\s*[=:]*\s*(.+)", p_text)
                    if time_match:
                        completion_time = time_match.group(1).strip()

    return {
        "Description": description,
        "Languages": languages,
        "Job_levels": job_levels,
        "Completion_time": completion_time,
    }


def update_csv_with_details(csv_path, html_dir_path, output_csv_path):
    """Process the CSV file and add descriptions, languages, job levels, and completion time from HTML files."""
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)
        rows = list(reader)

    new_columns = ["Description", "Languages", "Job_levels", "Completion_time"]
    for column in new_columns:
        if column not in headers:
            headers.append(column)

    for row in rows:
        product_name = row[0]
        product_url = row[headers.index("Link")]  # Get the URL from the Link column
        filename = sanitize_filename(product_url) + ".html"
        html_path = os.path.join(html_dir_path, filename)

        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as html_file:
                html_content = html_file.read()
                product_details = extract_product_details(html_content)

                for column in new_columns:
                    column_index = headers.index(column)
                    if len(row) <= column_index:
                        row.extend([""] * (column_index - len(row) + 1))
                    row[column_index] = (
                        product_details[column] if product_details[column] else "NULL"
                    )
        else:
            print(
                f"Warning: HTML file not found for {product_name} (expected: {filename})"
            )
            for column in new_columns:
                column_index = headers.index(column)
                if len(row) <= column_index:
                    row.extend([""] * (column_index - len(row) + 1))
                row[column_index] = "NULL"

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    csv_path = "/root/shl-assignment/backend/scraper/shl_solutions.csv"
    html_dir_path = "/root/shl-assignment/backend/scraper/shl_catalog_product_pages"
    output_csv_path = (
        "/root/shl-assignment/backend/scraper/shl_solutions_with_details.csv"
    )

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    update_csv_with_details(csv_path, html_dir_path, output_csv_path)

    print(f"CSV file updated with product details and saved to {output_csv_path}")
