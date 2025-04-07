import type { Recommendation } from "@/components/recommendation-table"

export function extractRecommendations(content: string): Recommendation[] {
  // Try to extract a markdown table from the content
  const tableRegex = /\|(.+)\|(.+)\|(.+)\|(.+)\|(.+)\|\n\|[\s-]+\|[\s-]+\|[\s-]+\|[\s-]+\|[\s-]+\|([\s\S]*?)(?:\n\n|$)/g
  const tableMatch = tableRegex.exec(content)

  if (!tableMatch) return []

  // Extract the table rows
  const tableContent = tableMatch[6]
  const rows = tableContent.trim().split("\n")

  // Parse each row into a recommendation object
  return rows
    .map((row) => {
      const cells = row.split("|").filter((cell) => cell.trim() !== "")

      if (cells.length < 5) return null

      return {
        name: cells[0].trim(),
        url: cells[1].trim(),
        remote_testing: cells[2].trim(),
        adaptive_irt: cells[3].trim(),
        test_type: cells[4].trim(),
      }
    })
    .filter(Boolean) as Recommendation[]
}

