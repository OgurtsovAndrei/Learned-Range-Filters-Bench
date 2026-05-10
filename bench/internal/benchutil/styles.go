package benchutil

// SeriesStyle describes plot appearance for a single series.
type SeriesStyle struct {
	Name   string
	Color  string
	Marker string
	Dashed bool
}

// DefaultSeriesStyles is the unified style set used across all plots.
var DefaultSeriesStyles = map[string]SeriesStyle{
	"Theoretical":   {Name: "Theoretical", Color: "#374151", Dashed: true, Marker: "circle"},
	"Grafite":       {Name: "Grafite", Color: "#0f766e", Marker: "diamond"},
	"Grafite-tuned": {Name: "Grafite-tuned", Color: "#14b8a6", Marker: "square"},
	"SNARF":         {Name: "SNARF", Color: "#1e3a8a", Marker: "diamond"},
	"SuRFReal(8)":   {Name: "SuRFReal(8)", Color: "#dc2626", Marker: "diamond"},
	"SODA":          {Name: "SODA", Color: "#ca8a04", Marker: "circle"},
	"Scan-ARE":      {Name: "Scan-ARE", Color: "#d946ef", Marker: "circle"},
	"Scan-ARE-Trunc":     {Name: "Scan-ARE-Trunc", Color: "#d946ef", Marker: "circle"},
	"Scan-ARE-SODA":      {Name: "Scan-ARE-SODA", Color: "#a21caf", Marker: "triangle"},
	"Greedy+Merge":       {Name: "Greedy+Merge", Color: "#ea580c", Marker: "circle"},
	"Greedy+Merge-Trunc": {Name: "Greedy+Merge-Trunc", Color: "#ea580c", Marker: "circle"},
	"Greedy+Merge-SODA":  {Name: "Greedy+Merge-SODA", Color: "#9a3412", Marker: "triangle"},
	"BloomARE":           {Name: "BloomARE", Color: "#9ca3af", Dashed: true, Marker: "circle"},
	
	// Legacy / internal filters.
	"Adaptive(t=0)": {Name: "Adaptive(t=0)", Color: "#2a7fff", Marker: "square"},
	"Hybrid":        {Name: "Hybrid", Color: "#9b59b6", Marker: "star"},
	"Truncation":    {Name: "Truncation", Color: "#e6a800", Marker: "triangle"},
	"CDF-ARE":       {Name: "CDF-ARE", Color: "#6366f1", Marker: "circle"},
	"SuRFNone":      {Name: "SuRFNone", Color: "#fca5a5", Marker: "diamond"},
	"SuRFHash":      {Name: "SuRFHash", Color: "#f87171", Marker: "diamond"},
	"SuRFReal":      {Name: "SuRFReal", Color: "#dc2626", Marker: "diamond"},
	"Rosetta":       {Name: "Rosetta", Color: "#15803d", Marker: "diamond"},
}
