/**
 * i18n — Language switching
 * Translates are embedded inline via a global variable SITE_I18N
 */
(function() {
  'use strict';

  const STORAGE_KEY = 'site_lang';

  function applyLanguage(lang) {
    if (!window.SITE_I18N || !window.SITE_I18N[lang]) return;
    document.documentElement.lang = lang;

    const data = window.SITE_I18N[lang];

    // Elements with data-i18n attribute (plain text replacement)
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      const text = key.split('.').reduce((acc, k) => acc && acc[k], data);
      if (text) {
        el.textContent = text;
      }
    });

    // Elements with data-i18n-html attribute (HTML content, e.g. bio with links)
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      const key = el.getAttribute('data-i18n-html');
      const html = key.split('.').reduce((acc, k) => acc && acc[k], data);
      if (html) {
        el.innerHTML = html;
      }
    });

    // Update toggle button text
    const toggle = document.getElementById('langToggle');
    if (toggle) {
      toggle.textContent = lang === 'en' ? '中' : 'EN';
    }

    localStorage.setItem(STORAGE_KEY, lang);
  }

  function init() {
    const saved = localStorage.getItem(STORAGE_KEY);
    const browserLang = navigator.language.startsWith('zh') ? 'zh' : 'en';
    const defaultLang = saved || browserLang;

    const toggle = document.getElementById('langToggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        const current = document.documentElement.lang || 'en';
        applyLanguage(current === 'en' ? 'zh' : 'en');
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
