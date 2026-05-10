package utils

// SeriesStyle describes plot appearance for a single series.
type SeriesStyle struct {
	Name   string
	Color  string
	Marker string
	Dashed bool
}

// DefaultSeriesStyles is the unified 8-series set used on FPR-vs-BPK plots.
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
}
