"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"

export interface Recommendation {
  name: string
  url: string
  remote_testing: string
  adaptive_irt: string
  test_type: string
}

interface RecommendationTableProps {
  recommendations: Recommendation[]
}

export function RecommendationTable({ recommendations }: RecommendationTableProps) {
  if (!recommendations.length) return null

  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900 overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-zinc-800/50">
            <TableHead className="w-[200px]">Assessment</TableHead>
            <TableHead className="hidden md:table-cell">Test Type</TableHead>
            <TableHead className="hidden md:table-cell">Remote</TableHead>
            <TableHead className="hidden md:table-cell">Adaptive</TableHead>
            <TableHead className="text-right">Link</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {recommendations.map((rec, index) => (
            <TableRow key={index} className="hover:bg-zinc-800/50">
              <TableCell className="font-medium">{rec.name}</TableCell>
              <TableCell className="hidden md:table-cell">{rec.test_type}</TableCell>
              <TableCell className="hidden md:table-cell">{rec.remote_testing}</TableCell>
              <TableCell className="hidden md:table-cell">{rec.adaptive_irt}</TableCell>
              <TableCell className="text-right">
                {rec.url && (
                  <Button variant="ghost" size="sm" className="h-8 w-8 p-0" asChild>
                    <a href={rec.url} target="_blank" rel="noopener noreferrer">
                      <ExternalLink className="h-4 w-4" />
                      <span className="sr-only">Open link</span>
                    </a>
                  </Button>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

