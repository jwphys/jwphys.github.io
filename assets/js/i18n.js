/**
 * i18n — Language switching for Jekyll site
 * Loads _data/en.yml and _data/zh.yml and replaces textContent
 * for elements with [data-i18n] attributes.
 */
(function() {
  'use strict';

  const STORAGE_KEY = 'site_lang';

  // Fetch YAML data files. We inline the translations from Jekyll.
  // Since we can't dynamically fetch YAML on GH Pages, we embed data in the page.
  let translations = {};

  function getAvailableLangs() {
    return ['en', 'zh'];
  }

  async function loadTranslations() {
    try {
      const [enRes, zhRes] = await Promise.all([
        fetch('/assets/i18n/en.json'),
        fetch('/assets/i18n/zh.json')
      ]);
      translations.en = await enRes.json();
      translations.zh = await zhRes.json();
    } catch (e) {
      console.warn('i18n: Could not fetch translation files, falling back to HTML defaults.');
      translations.en = {};
      translations.zh = {};
    }
  }

  function getValue(obj, path) {
    return path.split('.').reduce((acc, key) => acc && acc[key], obj);
  }

  function applyLanguage(lang) {
    if (!translations[lang]) return;
    document.documentElement.lang = lang;

    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      const text = getValue(translations[lang], key);
      if (text) {
        el.textContent = text;
      }
    });

    // Update lang toggle button text
    const toggle = document.getElementById('langToggle');
    if (toggle) {
      const nextLang = lang === 'en' ? '中' : 'EN';
      toggle.textContent = lang === 'en' ? '中' : 'EN';
    }

    localStorage.setItem(STORAGE_KEY, lang);
  }

  async function init() {
    await loadTranslations();

    const saved = localStorage.getItem(STORAGE_KEY);
    const browserLang = navigator.language.startsWith('zh') ? 'zh' : 'en';
    const defaultLang = saved || browserLang;

    const toggle = document.getElementById('langToggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        const current = document.documentElement.lang || 'en';
        const next = current === 'en' ? 'zh' : 'en';
        applyLanguage(next);
      });
    }

    applyLanguage(defaultLang);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
