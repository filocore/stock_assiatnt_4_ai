I will update the `gui` function in `stock_assistant.py` to:
1.  Restore the full data analysis logic (fetching K-lines, calculating indicators like MACD/RSI/Bollinger) by integrating the `fetch_and_store` pattern used in other modes.
2.  Ensure data is saved to `history.jsonl` for persistence, consistent with the rest of the application.
3.  Update the display to show detailed metrics (JSON format) in the details panel, matching the CLI output's richness.
4.  Maintain the watchlist functionality (add/delete) and auto-refresh features.
5.  Remove the unused `gui_qt` function to clean up the file.