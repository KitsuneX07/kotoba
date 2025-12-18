use std::collections::HashMap;
use std::time::Duration;

/// Extracts the `Retry-After` header (in seconds) if present.
///
/// Providers occasionally instruct clients to wait before re-sending requests. When the
/// header is numeric this helper parses it into a [`Duration`]. HTTP-date values are
/// currently ignored because vendors primarily use the numeric form.
pub(crate) fn retry_after_from_headers(headers: &HashMap<String, String>) -> Option<Duration> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))
        .and_then(|(_, value)| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}
