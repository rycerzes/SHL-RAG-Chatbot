import os
import csv
import glob
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def parse_shl_catalog_pages(directory_path):
    """
    Parse all HTML files in the specified directory to extract solution info
    """
    all_solutions = []
    base_url = "https://www.shl.com"

    html_files = glob.glob(os.path.join(directory_path, "*.html"))

    for html_file in html_files:
        print(f"Processing {os.path.basename(html_file)}...")

        with open(html_file, "r", encoding="utf-8") as file:
            content = file.read()

        soup = BeautifulSoup(content, "html.parser")

        # Find all custom__table-responsive divs
        table_divs = soup.find_all("div", class_="custom__table-responsive")

        for table_div in table_divs:
            # Determine the type of solution
            solution_type = None
            table = table_div.find("table")

            if table:
                heading = table.find("th", class_="custom__table-heading__title")
                if heading:
                    solution_type = heading.get_text(strip=True)

            if not solution_type:
                continue

            rows = table.find_all("tr")[1:]  # Skip header row

            for row in rows:
                solution_data = {"type": solution_type}

                cells = row.find_all("td")
                if not cells:
                    continue

                title_cell = cells[0]
                title_link = title_cell.find("a")

                if title_link:
                    solution_data["name"] = title_link.get_text(strip=True)
                    solution_data["link"] = urljoin(
                        base_url, title_link.get("href", "")
                    )
                else:
                    solution_data["name"] = title_cell.get_text(strip=True)
                    solution_data["link"] = ""

                if len(cells) > 1:
                    remote_testing_cell = cells[1]
                    solution_data["remote_testing"] = (
                        "Yes"
                        if remote_testing_cell.find(
                            "span", class_="catalogue__circle -yes"
                        )
                        else "No"
                    )
                else:
                    solution_data["remote_testing"] = "Unknown"

                if len(cells) > 2:
                    adaptive_irt_cell = cells[2]
                    solution_data["adaptive_irt"] = (
                        "Yes"
                        if adaptive_irt_cell.find(
                            "span", class_="catalogue__circle -yes"
                        )
                        else "No"
                    )
                else:
                    solution_data["adaptive_irt"] = "Unknown"

                if len(cells) > 3:
                    solution_data["test_type"] = cells[3].get_text(strip=True)
                else:
                    solution_data["test_type"] = "Unknown"

                all_solutions.append(solution_data)

    return all_solutions


def save_to_csv(solutions, output_file):
    """
    Save the extracted solution information to a CSV file
    """
    sorted_solutions = sorted(solutions, key=lambda x: x["name"].lower())

    headers = ["Name", "Link", "Type", "Remote Testing", "Adaptive/IRT", "Test Type"]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(headers)

        for solution in sorted_solutions:
            writer.writerow(
                [
                    solution.get("name", ""),
                    solution.get("link", ""),
                    solution.get("type", ""),
                    solution.get("remote_testing", ""),
                    solution.get("adaptive_irt", ""),
                    solution.get("test_type", ""),
                ]
            )

    print(f"CSV file saved: {output_file}")


def main():
    directory_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "shl_catalog_pages"
    )

    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "shl_solutions.csv"
    )

    solutions = parse_shl_catalog_pages(directory_path)

    save_to_csv(solutions, output_file)

    print(f"Total solutions extracted: {len(solutions)}")


if __name__ == "__main__":
    main()
