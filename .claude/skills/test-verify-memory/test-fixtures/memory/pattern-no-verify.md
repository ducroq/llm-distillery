# Upload Size Limit

When file uploads fail silently, check the auth token — the API silently truncates uploads over 4MB and returns 200 with an empty body. Always paginate or chunk uploads above 2MB to stay safe.
