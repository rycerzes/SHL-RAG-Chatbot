import asyncio
import random
import pandas as pd
from playwright.async_api import async_playwright
import os
import argparse
import time
import json


async def scrape_product_catalog(page_offsets, headless=True, slow_mo=100, debug=False):
    """Scrape SHL product catalog using Playwright"""
    products = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, slow_mo=slow_mo)

        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )

        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

        page = await context.new_page()

        page.set_default_timeout(60000)

        debug_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "debug"
        )
        if debug:
            os.makedirs(debug_dir, exist_ok=True)

        try:
            for offset in page_offsets:
                print(f"Scraping page with offset {offset}...")
                url = f"https://www.shl.com/solutions/products/product-catalog/?start={offset}&type=1"

                try:
                    response = await page.goto(
                        url, wait_until="networkidle", timeout=90000
                    )

                    if not response.ok:
                        print(
                            f"Failed to load page: {response.status} {response.status_text}"
                        )
                        continue

                    # ss for debugging
                    if debug:
                        await page.screenshot(
                            path=f"{debug_dir}/page_{offset}.png", full_page=True
                        )

                    # page HTML for debugging
                    html = await page.content()
                    if debug:
                        with open(
                            f"{debug_dir}/page_{offset}.html", "w", encoding="utf-8"
                        ) as f:
                            f.write(html)

                    print("Waiting for page content to load...")

                    selectors = [
                        ".custom_table-responsive",
                    ]

                    selector_found = False
                    for selector in selectors:
                        try:
                            print(f"Trying to find selector: {selector}")
                            await page.wait_for_selector(selector, timeout=10000)
                            print(f"Found selector: {selector}")
                            selector_found = True
                            break
                        except Exception as e:
                            print(f"Selector {selector} not found: {e}")

                    if not selector_found:
                        print("Could not find any expected content on the page.")
                        body_content = await page.inner_text("body")
                        print(f"Body content length: {len(body_content)} characters")
                        print(f"First 200 chars: {body_content[:200]}...")

                        element_counts = await page.evaluate("""() => {
                            const counts = {};
                            document.querySelectorAll('*').forEach(el => {
                                const tag = el.tagName.toLowerCase();
                                counts[tag] = (counts[tag] || 0) + 1;
                            });
                            return counts;
                        }""")
                        print("Elements found on page:")
                        print(json.dumps(element_counts, indent=2))

                        current_url = page.url
                        print(f"Current URL: {current_url}")
                        if current_url != url:
                            print(
                                "Redirected from original URL! Possible bot protection."
                            )

                        if debug:
                            await page.screenshot(
                                path=f"{debug_dir}/page_{offset}_error.png",
                                full_page=True,
                            )

                        continue

                    print("Searching for product information...")

                    tables = await page.query_selector_all("table")
                    print(f"Found {len(tables)} tables on the page")

                    if tables:
                        for table_index, table in enumerate(tables):
                            print(f"Processing table {table_index + 1}/{len(tables)}")

                            rows = await table.query_selector_all(
                                "tr:not(:first-child)"
                            )
                            print(f"Found {len(rows)} rows in table {table_index + 1}")

                            header_row = await table.query_selector("tr:first-child")
                            if header_row:
                                header_cells = await header_row.query_selector_all("th")
                                header_texts = []
                                for cell in header_cells:
                                    text = await cell.inner_text()
                                    header_texts.append(text.strip())
                                print(f"Table headers: {header_texts}")

                            for row_index, row in enumerate(rows):
                                try:
                                    cells = await row.query_selector_all("td")

                                    if len(cells) >= 2:
                                        product_name = "Unknown"
                                        product_link = ""
                                        product_data = {}

                                        first_cell = cells[0]
                                        product_link_elem = (
                                            await first_cell.query_selector("a")
                                        )

                                        if product_link_elem:
                                            product_name = (
                                                await product_link_elem.inner_text()
                                            )
                                            product_link = (
                                                await product_link_elem.get_attribute(
                                                    "href"
                                                )
                                            )

                                            if (
                                                product_link
                                                and not product_link.startswith("http")
                                            ):
                                                product_link = f"https://www.shl.com{product_link if product_link.startswith('/') else '/' + product_link}"
                                        else:
                                            product_name = await first_cell.inner_text()

                                        product_data["Product Name"] = (
                                            product_name.strip()
                                        )
                                        product_data["Product Link"] = product_link

                                        for i in range(1, len(cells)):
                                            cell_text = await cells[i].inner_text()
                                            column_name = (
                                                header_texts[i]
                                                if i < len(header_texts)
                                                else f"Column {i + 1}"
                                            )
                                            product_data[column_name] = (
                                                cell_text.strip()
                                            )

                                        products.append(product_data)
                                except Exception as e:
                                    print(
                                        f"Error processing row {row_index} in table {table_index}: {e}"
                                    )

                    print(f"Found {len(products)} products so far")

                except Exception as e:
                    print(f"Error processing page with offset {offset}: {e}")

                    # ss for debugging
                    if debug:
                        await page.screenshot(
                            path=f"{debug_dir}/error_page_{offset}.png", full_page=True
                        )

                # Add random delay between pages
                if offset < page_offsets[-1]:
                    delay = random.uniform(5, 10)  # Longer delay to avoid detection
                    print(f"Waiting {delay:.1f} seconds before next request...")
                    await asyncio.sleep(delay)

        finally:
            await browser.close()

    return products


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Scrape SHL product catalog using Playwright"
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Run in visible browser mode (not headless)",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=100,
        help="Slow down actions by this many milliseconds",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (screenshots and HTML dumps)",
    )
    parser.add_argument("--start", type=int, default=0, help="Starting offset")
    parser.add_argument("--end", type=int, default=372, help="Ending offset")
    parser.add_argument("--step", type=int, default=12, help="Step size for pagination")
    args = parser.parse_args()

    # Generate page offsets based on args
    page_offsets = list(range(args.start, args.end + 1, args.step))

    print("Starting SHL catalog scraper...")
    print(f"Mode: {'Visible browser' if args.visible else 'Headless'}")
    print(f"Will process {len(page_offsets)} pages")

    start_time = time.time()

    # Scrape products
    all_products = await scrape_product_catalog(
        page_offsets=page_offsets,
        headless=not args.visible,
        slow_mo=args.slow,
        debug=args.debug,
    )

    elapsed_time = time.time() - start_time
    print(f"Scraping completed in {elapsed_time:.1f} seconds")

    if all_products:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

        # Save to CSV and Excel
        print(f"Saving {len(all_products)} products to files...")
        df = pd.DataFrame(all_products)

        csv_path = os.path.join(output_dir, "shl_products.csv")
        excel_path = os.path.join(output_dir, "shl_products.xlsx")

        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)

        print(f"Successfully scraped {len(all_products)} products")
        print(f"Data saved to {csv_path} and {excel_path}")
    else:
        print("No products were found.")
        print("Try running with --visible and --debug flags to troubleshoot.")


if __name__ == "__main__":
    asyncio.run(main())
