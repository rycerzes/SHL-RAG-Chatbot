import type { Recommendation } from "@/components/recommendation-table"

export function extractRecommendations(content: string): Recommendation[] {
  // Check if the content contains a markdown table
  if (!content.includes("| --- |")) {
    return []
  }

  try {
    // Extract the table from the content
    const tableRegex = /\|(.+)\|[\s\S]+?\|([\s-]+\|)+[\s\S]+?(?=\n\n|\n$|$)/g
    const tableMatch = content.match(tableRegex)
    
    if (!tableMatch) return []
    
    const tableContent = tableMatch[0]
    const lines = tableContent.split('\n').filter(line => line.trim() !== '')
    
    // If we don't have at least header, separator, and one data row
    if (lines.length < 3) return []
    
    // Extract headers
    const headers = lines[0].split('|')
      .map(h => h.trim())
      .filter(h => h)
    
    // Skip the header and separator lines
    const dataRows = lines.slice(2)
    
    return dataRows.map(row => {
      const cells = row.split('|')
        .map(cell => cell.trim())
        .filter(cell => cell !== '')
      
      // Map cells to their corresponding headers
      const nameIndex = headers.findIndex(h => h.toLowerCase().includes('name') || h.toLowerCase().includes('assessment'))
      const urlIndex = headers.findIndex(h => h.toLowerCase().includes('url'))
      const remoteIndex = headers.findIndex(h => h.toLowerCase().includes('remote'))
      const adaptiveIndex = headers.findIndex(h => h.toLowerCase().includes('adaptive'))
      const typeIndex = headers.findIndex(h => h.toLowerCase().includes('type'))
      
      return {
        name: cells[nameIndex >= 0 ? nameIndex : 0] || '',
        url: cells[urlIndex >= 0 ? urlIndex : 1] || '',
        remote_testing: cells[remoteIndex >= 0 ? remoteIndex : 2] || '',
        adaptive_irt: cells[adaptiveIndex >= 0 ? adaptiveIndex : 3] || '',
        test_type: cells[typeIndex >= 0 ? typeIndex : 4] || ''
      }
    })
  } catch (error) {
    console.error('Error extracting recommendations:', error)
    return []
  }
}