This project automatically analyzes the top 50 life science stocks and generates a trading report with buy/sell scores, a summary PDF, and a visual chart. All results are published to a simple web page via GitHub Pages.

name: Upload site folder as artifact
        uses: actions/upload-artifact@v4
        with:
          name: site
          path: site
