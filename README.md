# Indian METAR Maps

Runs the METAR scraper every 30 minutes and publishes PNG maps with GitHub Pages.

Enable **Settings → Pages → Source: GitHub Actions**, then run the workflow manually once.

Website: `https://YOUR_USERNAME.github.io/YOUR_REPOSITORY/`

Direct PNG example: `https://YOUR_USERNAME.github.io/YOUR_REPOSITORY/metar_temperature_contours.png`

GitHub Actions uses UTC cron. The source endpoint must remain available to GitHub-hosted runners.
