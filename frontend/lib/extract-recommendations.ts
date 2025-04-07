import { Recommendation } from "@/components/recommendation-table"

export function extractRecommendations(text: string): Recommendation[] {
  // Look for markdown tables in the content
  const tableRegex = /\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|/g
  const headerRegex = /\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|/

  // If table header separator doesn't exist, return empty array
  if (!headerRegex.test(text)) {
    return []
  }

  // Find all rows (excluding the header separator row)
  const rows = [...text.matchAll(tableRegex)].filter(match => {
    // Filter out header separator row (contains dashes)
    return !match[0].includes("---") && !match[0].includes("| ---");
  });

  // Skip the header row (first row after filtering)
  const recommendations: Recommendation[] = []
  
  // Start from index 1 to skip header
  for (let i = 1; i < rows.length; i++) {
    const [_, name, url, remote_testing, adaptive_irt, test_type] = rows[i]
    
    // Clean up the URL (if it's a markdown link)
    const cleanUrl = url.trim().replace(/\[(.+?)\]\((.+?)\)/, "$2")
    
    recommendations.push({
      name: name.trim(),
      url: cleanUrl,
      remote_testing: remote_testing.trim(),
      adaptive_irt: adaptive_irt.trim(),
      test_type: test_type.trim()
    })
  }

  return recommendations
}