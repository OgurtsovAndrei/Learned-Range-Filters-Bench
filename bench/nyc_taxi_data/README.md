# NYC TLC Trip Record Datasets

Trip records published by the **NYC Taxi & Limousine Commission**. Public domain.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
CDN: https://d37ci6vzurychx.cloudfront.net/trip-data/

## Download

```bash
./download.sh
```

Skip files already on disk. Set `NYC_TAXI_PARALLEL=N` to change parallelism (default 2 — CloudFront in front of `nyc-tlc` rate-limits bursts; keep this low).

If many files start failing with HTTP 403 mid-run, the CDN has banned your IP for ~6–24h. Wait and resume; the script is idempotent.

## Services & coverage

| Service | First month | Last month  | Months  | Notes                                           |
|---------|-------------|-------------|---------|-------------------------------------------------|
| yellow  | 2009-01     | 2025-12     | ~204    | Classic yellow medallion taxis                  |
| green   | 2014-12     | 2025-12     | ~133    | Boro taxis (outer-borough green cabs)           |
| fhv     | 2015-01     | 2025-12     | ~132    | For-Hire Vehicles (early Uber/Lyft, livery)     |
| fhvhv   | 2019-12     | 2025-12     | ~73     | High-Volume FHV (Uber/Lyft post-regulation)     |

**Total ≈ 542 monthly parquet files, ≈ 25 GB on disk.**

## Format

Each file is a Parquet column store. Pickup-timestamp column name varies by service:

| Service | Pickup-timestamp column     |
|---------|------------------------------|
| yellow  | `tpep_pickup_datetime`       |
| green   | `lpep_pickup_datetime`       |
| fhv     | `pickup_datetime`            |
| fhvhv   | `pickup_datetime`            |

(Pre-2015 yellow parquet files use `pickup_datetime` without the TPEP prefix; the
reader handles both.)

## Read in Go

Read directly via `bench/datasets/nyc_taxi.go` — no intermediate format
conversion. Each `NYCTaxiPickupReader` wraps one or more parquet files and
yields sorted-unique `[]uint64` keys (Unix nanoseconds).
