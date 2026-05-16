# Jiwen Chen — Personal Academic Website

Built with Jekyll. Hosted on GitHub Pages.

## Local Development

```bash
gem install bundler
bundle install
bundle exec jekyll serve
```

## GitHub Pages Setup

1. Create repo: `jwphys.github.io` under the `jwphys` account
2. Push this repo to GitHub
3. Go to Settings → Pages → Source: Deploy from `main` branch, `/docs` folder or root
4. Site will be live at `https://jwphys.github.io`

## Updating Publications

Publications are stored in `_data/publications.yml`. Edit manually, or run the GitHub Action (requires adding `GOOGLE_SCHOLAR_ID` to repo secrets).

## Language Toggle

Translations are in `_data/en.yml` and `_data/zh.yml`. The JS in `assets/js/i18n.js` handles switching.
