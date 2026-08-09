# API rate limits

The public API allows 500 requests per minute per API key. Short bursts of up to
750 requests per minute are tolerated for no more than 30 seconds, after which
the limiter returns HTTP 429 with a `Retry-After` header.

Rate limits are applied per key, not per IP address. A key shared across several
services therefore shares one budget, which is the most common cause of
unexplained throttling in multi-service deployments.

## Raising a limit

Limit increases are reviewed weekly. Requests must include the peak sustained
throughput observed over the previous 30 days and the business justification.
Increases above 2000 requests per minute require a dedicated capacity review.
