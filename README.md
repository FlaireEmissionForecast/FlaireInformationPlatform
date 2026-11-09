# FlaireInformationPlatform
An SQLite database, FastAPI and charting website for serving forecast data to consumers.

To host the server on localhost, run in root dir:
uvicorn --app-dir . api.forecast_api:api --reload

Then tunnel or host the API wherever.

NEW: Now has an endpoint for querying ENTSO-E day-ahead spot prices for any bidding zone, with or without history.